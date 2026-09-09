import pytest

from igfold.utils.checkpoint import find_weights

HEAVY = "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS"
LIGHT = (
    "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
)
NANOBODY = "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS"


def weights_available() -> bool:
    try:
        find_weights()
        return True
    except FileNotFoundError:
        return False


requires_weights = pytest.mark.skipif(not weights_available(), reason="pre-trained IgFold weights not installed")


@pytest.fixture(scope="session")
def runner():
    from igfold import IgFoldRunner

    return IgFoldRunner(num_models=1, device="cpu", verbose=False)
