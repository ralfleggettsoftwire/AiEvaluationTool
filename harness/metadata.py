import httpx

_IMDS_INSTANCE_TYPE_URL = "http://169.254.169.254/latest/meta-data/instance-type"
_IMDS_TIMEOUT = 1.0


async def _fetch_model_name(endpoint_url: str, api_key: str | None) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            resp = await client.get(f"{endpoint_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return str(data["data"][0]["id"])
    except Exception as exc:
        raise RuntimeError(
            f"Could not detect model name from {endpoint_url}/v1/models — "
            "ensure the vLLM server is running and reachable."
        ) from exc


async def _fetch_hardware() -> str:
    try:
        async with httpx.AsyncClient(timeout=_IMDS_TIMEOUT) as client:
            resp = await client.get(_IMDS_INSTANCE_TYPE_URL)
            resp.raise_for_status()
            return resp.text.strip()
    except Exception as exc:
        raise RuntimeError(
            "Could not detect EC2 instance type from the instance metadata service — "
            "the harness must run on an EC2 instance."
        ) from exc


async def fetch_run_metadata(endpoint_url: str, *, api_key: str | None = None) -> tuple[str, str]:
    """Return (model_name, hardware) via vLLM /v1/models and EC2 IMDS."""
    model_name = await _fetch_model_name(endpoint_url, api_key)
    hardware = await _fetch_hardware()
    return model_name, hardware
