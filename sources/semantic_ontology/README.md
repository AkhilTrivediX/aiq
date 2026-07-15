# semantic-ontology-query

A NAT plugin that adds Semantic Ontology as a source in AI-Q.

The `semantic_ontology_query` tool asks the Semantic Ontology assistant a natural-language question via its
chat API (`POST /api/chat/completions`) and returns the final answer. Semantic Ontology
generates and runs SQL over internal structured datasets, then explains the
result.

## Authentication

AI-Q and Semantic Ontology authenticate against the **same NVIDIA SSO provider**. The tool
calls `aiq_agent.auth.get_auth_token()` to obtain the current user's SSO bearer
token and forwards it as `Authorization: Bearer <token>`. Semantic Ontology validates the
token against NVIDIA's JWKS before answering — no service account or shared
secret is involved, so every call is attributed to the signed-in user.

## Configuration

```yaml
functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: semantic_ontology
        name: "Semantic Ontology Assistant"
        description: "Ask the Semantic Ontology assistant about internal structured data."
        requires_auth: true
        tools:
          - semantic_ontology_query

  semantic_ontology_query:
    _type: semantic_ontology_query
    base_url: http://host.docker.internal:3100
```

`base_url` points at the Semantic Ontology **frontend**, which validates the bearer token and
proxies to the Semantic Ontology backend. In the local dev setup Semantic Ontology is exposed via a minikube
port-forward bound to `0.0.0.0:3100`, reachable from the AI-Q container at
`host.docker.internal:3100`.
