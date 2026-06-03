# OpenShift Overlay Guidelines

1. We introduced an `openshift` flag in the pre-existing [`values.yaml`](helm-charts-k8s/aiq/values.yaml) file. When `openshift.enabled` is `false` (the default), no OpenShift resources are rendered and the chart behaves identically to upstream.
2. All values and resources required for OpenShift are placed in dedicated files: [`values-openshift.yaml`](deployment-k8s/values-openshift.yaml) and [`openshift.yaml`](helm-charts-k8s/aiq/templates/openshift.yaml).
3. Secrets (NGC image-pull secret, API credentials) are created declaratively by the chart when the corresponding values are provided via `--set` flags. This keeps the deployment to a single `helm install` command with no manual `oc create secret` steps.
4. Container-level `securityContext` is made overridable per-app in [`deployment.yaml`](helm-charts-k8s/aiq/templates/deployment.yaml) so OpenShift's SCC can manage it (set to `null` in the overlay).
5. The main goal is to touch the original repository as little as possible, providing OpenShift deployment support with minimal changes.