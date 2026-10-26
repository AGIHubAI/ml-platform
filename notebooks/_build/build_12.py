"""Generator for notebooks/12_gpu_and_kubernetes_resource_orchestration.ipynb."""

from nbhelpers_b import code, main, md, preamble

NAME = "12_gpu_and_kubernetes_resource_orchestration"

cells = [
    md(
        r"""
        # 12 · GPUs, Kubernetes and resource orchestration

        Locally, `ray.init()` starts a one-node Ray cluster and every worker is a process on the laptop.
        In production the same code runs on a **Ray cluster that lives on Kubernetes** (via the KubeRay
        operator), with CPU and GPU node pools, autoscaling and shared storage. Nothing in
        `ml_platform` changes — only *where* Ray's workers come from.

        This notebook (no Kubernetes cluster needed):

        1. defines the Kubernetes vocabulary the platform relies on;
        2. loads and inspects the repository's manifests in `infra/k8s/`;
        3. simulates the Kubernetes scheduler's filter/score steps on those pods to show node pools,
           selectors, affinity, taints and tolerations at work;
        4. runs the platform's **resource selection** logic (`ray_runtime.resources`) on this machine.
        """
    ),
    md(
        r"""
        ## Conceptual model

        | Kubernetes object | What it is | In this platform |
        |---|---|---|
        | **Node** | a VM / machine with allocatable CPU, memory, GPUs; labels + taints | nodes in pools `system`, `cpu`, `cpu-ondemand`, `gpu` (label `ml.io/pool`) |
        | **Pod** | one or more containers scheduled together on one node; unit of scheduling | a Ray head or Ray worker process group |
        | **requests / limits** | requests = what the scheduler reserves; limits = what the kernel enforces | Ray derives its logical `CPU`/`GPU`/`memory` from the pod's resources |
        | **Deployment** | keeps N identical pods running (stateless services) | MLflow server, Dagster webserver/daemon (`dagster-values.yaml`) |
        | **Job** | runs pods to completion, with retries (`backoffLimit`) | the RayJob submitter pod |
        | **node pool** | a group of identical nodes (same instance type), scaled together | spot CPU pool, on-demand CPU pool, GPU pool |
        | **nodeSelector / affinity** | hard / soft constraints on *which nodes* a pod may use | GPU workers → `ml.io/pool: gpu`; CPU workers *prefer* spot |
        | **taint / toleration** | a node repels pods unless they tolerate the taint | GPU nodes tainted `nvidia.com/gpu:NoSchedule` so CPU pods never occupy them |
        | **extended resource** `nvidia.com/gpu` | whole GPUs advertised by the device plugin; requests must equal limits | GPU worker pods request `nvidia.com/gpu: 1` |
        | **autoscaling** | HPA (pods), Cluster Autoscaler / Karpenter (nodes) | Serve autoscaler → KubeRay autoscaler → node autoscaler |
        | **PersistentVolumeClaim** | durable storage outliving pods; `ReadWriteMany` = shared by many nodes | `ml-shared` for rounds, reports, checkpoints |
        | **KubeRay CRDs** | `RayCluster`, `RayJob`, `RayService` — the operator turns them into pods/services | `raycluster.yaml`, `rayjob.yaml`, `rayservice.yaml` |

        ```text
          Dagster (Deployment) ──ray://head:10001──►  RayCluster (KubeRay)
                                                       ├─ head pod        (pool: system, num-cpus 0: GCS, dashboard, Serve controller)
                                                       ├─ worker group cpu (pool: cpu, prefers spot, 1..10 pods × 4 CPU)
                                                       └─ worker group gpu (pool: gpu, tolerates GPU taint, 0..4 pods × 1 GPU)
          Ray Serve replicas / Ray Train workers / Ray tasks  ── are placed by *Ray* onto these pods
          pods ── are placed by *Kubernetes* onto nodes;  nodes ── are created by the *node autoscaler*
        ```

        Three schedulers, three levels: the node autoscaler provisions machines, Kubernetes places
        pods on machines, Ray places tasks/actors/Train workers into pods. Dagster sits above all of
        them and decides *what* should be computed.
        """
    ),
    preamble("nb12"),
    code(
        r"""
        import re
        import yaml
        from ml_platform.config import REPO_ROOT

        K8S = REPO_ROOT / "infra" / "k8s"
        docs = {}
        for f in sorted(K8S.glob("*.yaml")):
            for d in yaml.safe_load_all(f.read_text()):
                if isinstance(d, dict) and "kind" in d:
                    docs[(f.name, d["kind"], d["metadata"]["name"])] = d
        pd.DataFrame([{"file": f, "kind": k, "name": n,
                       "apiVersion": d.get("apiVersion")} for (f, k, n), d in docs.items()])
        """
    ),
    md(
        r"""
        ## 1 · What the manifests ask for

        Resource quantities use Kubernetes units: CPU in cores or millicores (`500m` = 0.5 core), memory
        in binary units (`8Gi`). A pod whose requests equal its limits for every container gets the
        **Guaranteed** QoS class (last to be evicted, and its CPU budget matches what Ray believes it
        has) — the manifests do this deliberately for Ray pods.
        """
    ),
    code(
        r"""
        def cpu(q):
            q = str(q); return float(q[:-1]) / 1000 if q.endswith("m") else float(q)

        def mem_gib(q):
            m = re.fullmatch(r"([\d.]+)(Ki|Mi|Gi|Ti)?", str(q))
            return float(m.group(1)) * {"Ki": 2**-20, "Mi": 2**-10, "Gi": 1, "Ti": 1024, None: 2**-30}[m.group(2)]

        def groups(cluster_spec):
            yield "head", None, cluster_spec["headGroupSpec"]
            for g in cluster_spec.get("workerGroupSpecs", []):
                yield g["groupName"], g, g

        def pod_row(source, gname, g, spec):
            pod = spec["template"]["spec"]
            c = pod["containers"][0]
            req, lim = c.get("resources", {}).get("requests", {}), c.get("resources", {}).get("limits", {})
            params = spec.get("rayStartParams", {})
            qos = "Guaranteed" if req and req == lim else ("Burstable" if req else "BestEffort")
            return {"manifest": source, "group": gname,
                    "replicas (min..max)": "1" if g is None else f"{g.get('minReplicas')}..{g.get('maxReplicas')}",
                    "cpu req/lim": f"{req.get('cpu')}/{lim.get('cpu')}",
                    "mem req/lim": f"{req.get('memory')}/{lim.get('memory')}",
                    "gpu": lim.get("nvidia.com/gpu", 0),
                    "QoS": qos,
                    "Ray CPUs": float(params["num-cpus"]) if "num-cpus" in params else cpu(lim.get("cpu", 0)),
                    "nodeSelector": ",".join(f"{k}={v}" for k, v in pod.get("nodeSelector", {}).items()),
                    "tolerations": ",".join(t["key"].split("/")[-1] for t in pod.get("tolerations", [])),
                    "affinity": "preferred" if pod.get("affinity") else ""}

        rows = []
        cluster_specs = {
            "raycluster.yaml": docs[("raycluster.yaml", "RayCluster", "ml-ray")]["spec"],
            "rayjob.yaml": docs[("rayjob.yaml", "RayJob", "ml-train")]["spec"]["rayClusterSpec"],
            "rayservice.yaml": docs[("rayservice.yaml", "RayService", "ml-serve")]["spec"]["rayClusterConfig"],
        }
        for src, cs in cluster_specs.items():
            for gname, g, spec in groups(cs):
                rows.append(pod_row(src, gname, g, spec))
        pods_df = pd.DataFrame(rows)
        pods_df
        """
    ),
    code(
        r"""
        rc = docs[("raycluster.yaml", "RayCluster", "ml-ray")]["spec"]
        cap = []
        for label, pick in (("min", "minReplicas"), ("initial", "replicas"), ("max", "maxReplicas")):
            tot = {"CPU": 0.0, "GPU": 0.0, "memory GiB": 0.0}
            for gname, g, spec in groups(rc):
                n = 1 if g is None else g[pick]
                lim = spec["template"]["spec"]["containers"][0]["resources"]["limits"]
                ray_cpu = float(spec["rayStartParams"].get("num-cpus", cpu(lim["cpu"])))
                tot["CPU"] += n * ray_cpu
                tot["GPU"] += n * float(lim.get("nvidia.com/gpu", 0))
                tot["memory GiB"] += n * mem_gib(lim["memory"])
            cap.append({"scale": label, **tot})
        print("ml-ray RayCluster: Ray logical resources at each scale (head contributes 0 CPUs: num-cpus=0)")
        print("autoscaler:", rc["autoscalerOptions"]["idleTimeoutSeconds"], "s idle timeout;",
              "enableInTreeAutoscaling =", rc["enableInTreeAutoscaling"])
        pd.DataFrame(cap).set_index("scale")
        """
    ),
    code(
        r"""
        pvc = docs[("raycluster.yaml", "PersistentVolumeClaim", "ml-shared")]
        job = docs[("rayjob.yaml", "RayJob", "ml-train")]["spec"]
        svc = docs[("rayservice.yaml", "RayService", "ml-serve")]["spec"]
        serve_cfg = yaml.safe_load(svc["serveConfigV2"])
        print("PVC ml-shared:", pvc["spec"]["accessModes"], pvc["spec"]["resources"]["requests"]["storage"],
              "storageClass", pvc["spec"]["storageClassName"])
        mounts = rc["headGroupSpec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        print("  mounted at:", [m["mountPath"] for m in mounts])
        print("\nRayJob ml-train:", {k: job[k] for k in ("entrypoint", "submissionMode", "shutdownAfterJobFinishes",
                                                           "backoffLimit", "activeDeadlineSeconds")})
        app = serve_cfg["applications"][0]
        print("\nRayService ml-serve app:", {k: app[k] for k in ("name", "import_path", "route_prefix")})
        print("  http:", serve_cfg["http_options"], "| proxy_location:", serve_cfg["proxy_location"])
        """
    ),
    md(
        r"""
        Things to notice:

        * The **head** asks Ray for `num-cpus: 0`: it hosts the GCS, dashboard and Serve controller but no
          tasks, so a runaway task cannot starve the control plane. It lives on the stable `system` pool —
          losing the head loses the cluster (unless GCS fault tolerance with external Redis is set up).
        * **CPU workers** prefer spot nodes (soft affinity + spot toleration): CPU work is retry-safe
          (Ray task retries, Ray Train resumes from checkpoints — notebook 11).
        * **GPU workers** use `minReplicas: 0` (no idle GPU bill), a hard `nodeSelector` for the GPU pool
          and tolerate its taint. `nvidia.com/gpu` must be set in *limits* (requests default to limits).
        * **Serve workers** in the RayService use a separate **on-demand** pool: latency-sensitive
          serving should not be preempted.
        * The **RayJob** creates an ephemeral, fixed-size cluster (`enableInTreeAutoscaling: false`,
          `shutdownAfterJobFinishes: true`); its `backoffLimit` retries the *whole job* — epoch-level
          recovery happens inside Ray Train, which is much cheaper.
        * The shared **PVC** is `ReadWriteMany` because Ray Train workers on different nodes must see
          the same checkpoint and round paths; object storage (`s3://…`) is the production alternative.

        ## 2 · How Kubernetes places these pods: filter → score → bind

        The kube-scheduler, for each pending pod: **filters** nodes (enough allocatable
        resources for the *requests*; `nodeSelector` labels match; every `NoSchedule` taint tolerated),
        **scores** the survivors (preferred affinity weights, spreading, packing…) and **binds** to the
        best. Below is a deliberately tiny re-implementation applied to the pods of `raycluster.yaml`
        (with the GPU group scaled to 1) on a toy inventory of nodes.
        """
    ),
    code(
        r"""
        nodes = [
            {"name": "sys-1",   "labels": {"ml.io/pool": "system"}, "taints": [], "cpu": 4, "mem": 16, "gpu": 0},
            {"name": "cpu-spot-1", "labels": {"ml.io/pool": "cpu", "karpenter.sh/capacity-type": "spot"},
             "taints": [("cloud.google.com/gke-spot", "true")], "cpu": 8, "mem": 32, "gpu": 0},
            {"name": "cpu-od-1", "labels": {"ml.io/pool": "cpu", "karpenter.sh/capacity-type": "on-demand"},
             "taints": [], "cpu": 8, "mem": 32, "gpu": 0},
            {"name": "gpu-1",   "labels": {"ml.io/pool": "gpu"}, "taints": [("nvidia.com/gpu", None)],
             "cpu": 16, "mem": 64, "gpu": 1},
        ]

        def pod_from(gname, spec):
            p = spec["template"]["spec"]; r = p["containers"][0]["resources"]["requests"]
            prefs = []
            for term in p.get("affinity", {}).get("nodeAffinity", {}).get("preferredDuringSchedulingIgnoredDuringExecution", []):
                for e in term["preference"]["matchExpressions"]:
                    prefs.append((term["weight"], e["key"], set(e["values"])))
            return {"pod": gname, "cpu": cpu(r["cpu"]), "mem": mem_gib(r["memory"]), "gpu": float(r.get("nvidia.com/gpu", 0)),
                    "selector": p.get("nodeSelector", {}), "tolerations": p.get("tolerations", []), "prefs": prefs}

        def tolerated(taint, tolerations):
            key, value = taint
            return any(t["key"] == key and (t.get("operator") == "Exists" or t.get("value") == value) for t in tolerations)

        def schedule(pod, nodes):
            reasons = {}
            feasible = []
            for n in nodes:
                if any(n["labels"].get(k) != v for k, v in pod["selector"].items()):
                    reasons[n["name"]] = "nodeSelector mismatch"; continue
                bad = [t[0] for t in n["taints"] if not tolerated(t, pod["tolerations"])]
                if bad:
                    reasons[n["name"]] = f"untolerated taint {bad[0]}"; continue
                if pod["cpu"] > n["cpu"] or pod["mem"] > n["mem"] or pod["gpu"] > n["gpu"]:
                    reasons[n["name"]] = "insufficient resources"; continue
                score = sum(w for w, k, vals in pod["prefs"] if n["labels"].get(k) in vals)
                feasible.append((score, n))
            if not feasible:
                return None, reasons
            score, best = max(feasible, key=lambda x: x[0])
            best["cpu"] -= pod["cpu"]; best["mem"] -= pod["mem"]; best["gpu"] -= pod["gpu"]
            return best["name"], reasons

        import copy
        inventory = copy.deepcopy(nodes)
        pending = [pod_from("head", rc["headGroupSpec"])]
        for g in rc["workerGroupSpecs"]:
            pending += [pod_from(f"{g['groupName']}-{i}", g) for i in range(max(g["replicas"], 1))]
        placement = []
        for pod in pending:
            node, why = schedule(pod, inventory)
            placement.append({"pod": pod["pod"], "requests": f"{pod['cpu']:g} CPU, {pod['mem']:g}Gi, {pod['gpu']:g} GPU",
                              "bound to": node or "PENDING", "filtered out": "; ".join(f"{k}: {v}" for k, v in why.items())})
        pd.set_option("display.max_colwidth", 200)
        pd.DataFrame(placement)
        """
    ),
    code(
        r"""
        # What if someone deletes the GPU toleration? -> the pod can never land on the GPU pool.
        broken = pod_from("gpu-no-toleration", rc["workerGroupSpecs"][1]); broken["tolerations"] = []
        print("gpu worker without toleration ->", schedule(broken, copy.deepcopy(nodes)))
        # And a CPU pod without a nodeSelector cannot take the GPU node either, thanks to the taint:
        stray = {"pod": "stray-cpu", "cpu": 4, "mem": 8, "gpu": 0, "selector": {}, "tolerations": [], "prefs": []}
        print("unconstrained CPU pod ->", schedule(stray, [copy.deepcopy(nodes[3])]))
        """
    ),
    md(
        r"""
        A pod that no node can fit stays **Pending**; that pending pod is exactly the signal the node
        autoscaler (Cluster Autoscaler / Karpenter) uses to add a node from the matching pool — and the
        KubeRay autoscaler creates such pods in the first place when *Ray* has pending resource demand
        (e.g. a `TorchTrainer` asking for 4 GPU workers). Idle pods are removed after
        `idleTimeoutSeconds`; then idle nodes are scaled down.

        ## 3 · Resource selection on *this* machine

        Ray has the same "pending, not failing" behaviour one level down: a task that requests a resource
        the cluster does not have is **not rejected** — it waits forever. That is why the platform never
        hard-codes `num_gpus` but chooses requests at run time
        (`ray_runtime.resources.select_resources`).
        """
    ),
    code(
        r"""
        import ray, torch
        from ml_platform.config import PlatformConfig
        from ml_platform.ray_runtime.cluster import ensure_ray
        from ml_platform.ray_runtime.resources import (available_cpus, cuda_gpu_count, gpu_matmul_benchmark,
                                                          run_accelerated_example, select_resources)

        ensure_ray(num_cpus=6, log_to_driver=False)
        res = ray.cluster_resources()
        print("Ray cluster resources:", {k: v for k, v in res.items() if not k.startswith("node:")})
        print(f"torch.cuda.is_available() = {torch.cuda.is_available()};  "
              f"cuda_gpu_count() = {cuda_gpu_count()};  available_cpus() = {available_cpus()}")
        """
    ),
    md(
        r"""
        On an Apple-silicon Mac, Ray advertises `GPU: 1` (Metal) plus an `accelerator_type:M…` label, but
        PyTorch DDP/NCCL needs **CUDA**; `cuda_gpu_count()` therefore reports 0 and the platform runs on
        CPU. On a Linux box with NVIDIA GPUs (or a KubeRay GPU pod) the same call returns the GPU count.
        """
    ),
    code(
        r"""
        cfg = PlatformConfig.for_tests(WORK)
        scenarios = {
            "default (2 workers, auto GPU)": cfg,
            "ask for 8 workers x 1 CPU": cfg.with_overrides(**{"distributed.num_workers": 8}),
            "ask for 4 workers x 4 CPUs": cfg.with_overrides(**{"distributed.num_workers": 4, "distributed.cpus_per_worker": 4}),
            "force use_gpu=true": cfg.with_overrides(**{"distributed.use_gpu": True}),
        }
        rows = []
        for name, c in scenarios.items():
            wr = select_resources(c)
            rows.append({"scenario": name, "num_workers": wr.num_workers, "use_gpu": wr.use_gpu,
                         "cpus/worker": wr.cpus_per_worker, "gpus/worker": wr.gpus_per_worker,
                         "ScalingConfig kwargs": wr.scaling_config_kwargs(), "notes": "; ".join(wr.notes)})
        pd.set_option("display.max_colwidth", 80)
        pd.DataFrame(rows).set_index("scenario")
        """
    ),
    code(
        r"""
        # A request the cluster cannot satisfy is queued, not rejected:
        impossible = gpu_matmul_benchmark.options(num_cpus=1, num_gpus=4).remote(64)
        ready, _ = ray.wait([impossible], timeout=3)
        print("task asking for 4 GPUs ready after 3 s?", bool(ready), "-> still pending; cancelling")
        ray.cancel(impossible, force=True)

        # The platform's pattern: static annotation on the function, dynamic override at call time.
        print(run_accelerated_example(cfg, n=256))
        """
    ),
    md(
        r"""
        ## 4 · Mapping local Ray onto Kubernetes

        | Local (this laptop) | Kubernetes / KubeRay |
        |---|---|
        | `ensure_ray()` starts a local head + raylet | `RayCluster` head pod; drivers connect with `RAY_ADDRESS=ray://ml-ray-head-svc:10001` (same `ensure_ray`, which reads `RAY_ADDRESS`) |
        | `ensure_ray(num_cpus=6)` caps logical CPUs | pod `limits.cpu` / `rayStartParams.num-cpus` define Ray's `CPU` per node |
        | worker *processes* | worker *pods* in worker groups (`cpu`, `gpu`, `serve-cpu`) |
        | `select_resources(cfg)` → `ScalingConfig` | same call; on GPU pods `cuda_gpu_count()` > 0 → `use_gpu=True`, one worker per GPU |
        | `train_distributed(...)` via `make train` | `RayJob` running `scripts/run_training.py` on an ephemeral cluster |
        | `serve.run(build_app(...))` via `make serve` | `RayService` with `import_path: ml_platform.inference.serve:app`, blue/green upgrades |
        | `data/`, `artifacts/`, `reports/` on local disk | `ReadWriteMany` PVC `ml-shared` (or object storage) mounted at `/app/...` |
        | `sqlite:///mlflow.db` | MLflow Deployment with Postgres + bucket artifact store |
        | `dagster dev` | Dagster Helm chart (`dagster-values.yaml`); assets call Ray through `RayComputeResource` |

        ## Connection to this repository

        * `infra/k8s/raycluster.yaml`, `rayjob.yaml`, `rayservice.yaml`, `dagster-values.yaml`, `infra/k8s/README.md`
        * `src/ml_platform/ray_runtime/cluster.py` → `ensure_ray` (local vs `RAY_ADDRESS`)
        * `src/ml_platform/ray_runtime/resources.py` → `select_resources`, `WorkerResources.scaling_config_kwargs`,
          `cuda_gpu_count` (Apple-Metal caveat, remote probe via `_probe_cuda`), `gpu_matmul_benchmark`,
          `run_accelerated_example`
        * `src/ml_platform/inference/batch.py` → `actor_resources`; `inference/serve.py` → `replica_resources`
          (per-actor / per-replica requests derived from what the cluster has)
        * `src/ml_platform/training/distributed.py` → `train_distributed` (Ray Train `ScalingConfig` from `select_resources`)

        ## Failure modes

        * **Pending forever** — requesting `num_gpus=1` on a CPU-only cluster (or a node pool at
          `maxReplicas`) queues silently. Always derive requests from the cluster, and alert on
          long-pending demand.
        * **Requests ≠ what Ray thinks** — Burstable pods whose Ray `num-cpus` exceeds the CPU request get
          throttled/evicted under contention; keep requests = limits for Ray pods.
        * **GPU nodes filled with CPU pods** — without taints, expensive GPU nodes run CPU work and GPU
          pods cannot be scheduled.
        * **Preemption of stateful components** — putting the Ray head or latency-critical Serve replicas
          on spot capacity.
        * **ReadWriteOnce storage** for multi-node training — only one node can mount it; checkpoints
          written by rank 0 are invisible to the others.
        * **Version skew** — the Ray version in the image must match `rayVersion` and the client's Ray.
        * **Scale-up latency** — node provisioning + image pull + model load can take minutes; keep
          `minReplicas` for online paths, and scale GPU groups from 0 only for batch work.

        ## Exercise

        1. Add a second GPU worker group `gpu-spot` that tolerates spot *and* GPU taints, with
           `maxReplicas: 8`, and extend the toy scheduler inventory with a spot GPU node. Which pods
           land where? What must be true of the training job for this to be safe (notebook 11)?
        2. Compute the Ray logical resources of `rayjob.yaml`'s cluster and check that they satisfy
           `select_resources(load_config("distributed"))`.
        3. Why is `nvidia.com/gpu` allowed only as a whole number in requests, while Ray allows
           `num_gpus=0.25`? What does fractional GPU sharing rely on, and what can go wrong?
        """
    ),
    code(
        r"""
        ray.shutdown()
        import shutil
        shutil.rmtree(WORK, ignore_errors=True)
        print(f"done in {time.perf_counter() - T0:.1f}s; Ray shut down, workspace removed")
        """
    ),
]

if __name__ == "__main__":
    main(NAME, cells)
