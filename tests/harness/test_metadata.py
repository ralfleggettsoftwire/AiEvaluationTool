import httpx
import pytest
import respx

from harness.metadata import fetch_run_metadata

BASE_URL = "http://fake-vllm"
IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_URL = "http://169.254.169.254/latest/meta-data/instance-type"
FAKE_TOKEN = "fake-imdsv2-token"


def _mock_imds(token: str = FAKE_TOKEN, instance_type: str = "g5.12xlarge") -> None:
    respx.put(IMDS_TOKEN_URL).mock(return_value=httpx.Response(200, text=token))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text=instance_type))


def _mock_models(model_id: str = "llama3") -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": model_id}]})
    )


@respx.mock
async def test_fetch_run_metadata_returns_model_name_and_instance_type() -> None:
    _mock_models("codellama-34b")
    _mock_imds(instance_type="g5.12xlarge")

    model_name, hardware = await fetch_run_metadata(BASE_URL)

    assert model_name == "codellama-34b"
    assert hardware == "g5.12xlarge"


@respx.mock
async def test_fetch_run_metadata_strips_whitespace_from_instance_type() -> None:
    _mock_models()
    _mock_imds(instance_type="  p4d.24xlarge\n")

    _, hardware = await fetch_run_metadata(BASE_URL)

    assert hardware == "p4d.24xlarge"


@respx.mock
async def test_fetch_run_metadata_sends_auth_header_when_api_key_set() -> None:
    models_route = respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama3"}]})
    )
    _mock_imds()

    await fetch_run_metadata(BASE_URL, api_key="secret-key")

    assert models_route.calls[0].request.headers["authorization"] == "Bearer secret-key"  # type: ignore[reportUnknownMemberType]


@respx.mock
async def test_imdsv2_token_forwarded_to_instance_type_request() -> None:
    _mock_models()
    token_route = respx.put(IMDS_TOKEN_URL).mock(
        return_value=httpx.Response(200, text="my-session-token")
    )
    instance_route = respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    await fetch_run_metadata(BASE_URL)

    assert token_route.called  # type: ignore[reportUnknownMemberType]
    sent_token = instance_route.calls[0].request.headers.get("x-aws-ec2-metadata-token")  # type: ignore[reportUnknownMemberType]
    assert sent_token == "my-session-token"


@respx.mock
async def test_fetch_run_metadata_raises_on_model_server_unreachable() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    _mock_imds()

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_model_server_http_error() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(return_value=httpx.Response(503))
    _mock_imds()

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_empty_models_list() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    _mock_imds()

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_token_failure() -> None:
    _mock_models()
    respx.put(IMDS_TOKEN_URL).mock(side_effect=httpx.ConnectError("no IMDS"))
    # GET should not be reached, but register it to avoid respx unmatched-route errors
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_unreachable() -> None:
    _mock_models()
    respx.put(IMDS_TOKEN_URL).mock(return_value=httpx.Response(200, text=FAKE_TOKEN))
    respx.get(IMDS_URL).mock(side_effect=httpx.ConnectError("no IMDS"))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_http_error() -> None:
    _mock_models()
    respx.put(IMDS_TOKEN_URL).mock(return_value=httpx.Response(200, text=FAKE_TOKEN))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_token_http_error() -> None:
    _mock_models()
    respx.put(IMDS_TOKEN_URL).mock(return_value=httpx.Response(401))
    respx.get(IMDS_URL).mock(return_value=httpx.Response(200, text="g5.2xlarge"))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_error_message_includes_endpoint_url() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    _mock_imds()

    with pytest.raises(RuntimeError, match=BASE_URL):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_missing_id_field_in_models_response() -> None:
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"name": "llama3"}]})
    )
    _mock_imds()

    with pytest.raises(RuntimeError, match="Could not detect model name"):
        await fetch_run_metadata(BASE_URL)


@respx.mock
async def test_fetch_run_metadata_raises_on_imds_timeout() -> None:
    _mock_models()
    respx.put(IMDS_TOKEN_URL).mock(return_value=httpx.Response(200, text=FAKE_TOKEN))
    respx.get(IMDS_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    with pytest.raises(RuntimeError, match="Could not detect EC2 instance type"):
        await fetch_run_metadata(BASE_URL)
