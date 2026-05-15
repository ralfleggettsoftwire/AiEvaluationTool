import httpx
import pytest
import respx

from harness.metadata import fetch_run_metadata

BASE_URL = "http://fake-vllm"
IMDS_URL = "http://169.254.169.254/latest/meta-data/instance-type"


@respx.mock
async def test_fetch_run_metadata_returns_model_name_and_instance_type() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "codellama-34b"}]})
    )
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.12xlarge"))

    model_name, hardware = await fetch_run_metadata(BASE_URL)

    assert model_name == "codellama-34b"
    assert hardware == "g5.12xlarge"


@respx.mock
async def test_fetch_run_metadata_strips_whitespace_from_instance_type() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="  p4d.24xlarge\n"))

    _, hardware = await fetch_run_metadata(BASE_URL)

    assert hardware == "p4d.24xlarge"


@respx.mock
async def test_fetch_run_metadata_sends_auth_header_when_api_key_set() -> None:
    models_route = respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    await fetch_run_metadata(BASE_URL, api_key="secret-key")

    assert models_route.calls[0].request.headers["authorization"] == "Bearer secret-key"  # type: ignore[reportUnknownMemberType]


@respx.mock
async def test_fetch_run_metadata_raises_on_model_server_unreachable() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_model_server_http_error() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(return_value=httpx.Response(503))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_empty_models_list() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_unreachable() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.get(IMDS_URL).mock(side_effect=httpx.ConnectError("no IMDS"))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_http_error() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.get(IMDS_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_error_message_includes_endpoint_url() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match=BASE_URL):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_missing_id_field_in_models_response() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"name": "llama3"}]})
    )
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_timeout() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    respx.get(IMDS_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)
