"""Every dataset the published corpus is built from.

To add a source: add it here with its pinned files and credit, then read it in train/corpus.py.
A changed upstream file fails its checksum and stops the release.
"""

from dataclasses import dataclass

SCB_REVISION = "4aa81b965a3e6abf1f97c83987cd18ddef8526b6"
SCB_FILES = f"https://huggingface.co/datasets/pythainlp/scb_mt_enth_2020/resolve/{SCB_REVISION}/data"
BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"
BY = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str
    path: str  # under data/public
    unpack: bool = False


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    creators: str
    homepage: str
    licence: str
    licence_url: str
    selection: str
    downloads: tuple


SOURCES = (
    Source(
        name="scb",
        title="SCB-MT-EN-TH-2020",
        creators="Lowphansirikul, Polpanumas, Rutherford and Nutanong, VISTEC and SCB",
        homepage="https://github.com/vistec-AI/dataset-releases",
        licence="CC BY-SA 4.0",
        licence_url=BY_SA,
        selection="Only the professionally translated task dialogs and generated reviews.",
        downloads=(
            Download(f"{SCB_FILES}/train-00000-of-00001.parquet", "c8ceca955df6ab5e4a2830f38e1c844436b0d78f10652def6292e8c8bbc4b28d", "scb/train.parquet"),
            Download(f"{SCB_FILES}/validation-00000-of-00001.parquet", "183e9a93bedaaf9ad720323b94a20e7464a1750089132759fc08ed3066af6aa7", "scb/validation.parquet"),
            Download(f"{SCB_FILES}/test-00000-of-00001.parquet", "762fc0250c2e74f5fbba53bd61e47e3be6bfceb4394588a62a207c76d890c1f2", "scb/test.parquet"),
        ),
    ),
    Source(
        name="mtop",
        title="MTOP",
        creators="Li, Arora, Chen, Gupta, Gupta and Mehdad, Meta",
        homepage="https://fb.me/mtop_dataset",
        licence="CC BY-SA 4.0",
        licence_url=BY_SA,
        selection="Thai utterances matched to their English originals by id, with the same slots in both.",
        downloads=(Download("https://dl.fbaipublicfiles.com/mtop/mtop.zip", "0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b", "mtop/mtop.zip", unpack=True),),
    ),
    Source(
        name="massive",
        title="MASSIVE 1.1, built on SLURP by Bastianelli, Vanzo, Swietojanski and Rieser",
        creators="FitzGerald and others, Amazon",
        homepage="https://github.com/alexa/massive",
        licence="CC BY 4.0",
        licence_url=BY,
        selection="Thai rows with unanimous top human ratings, no localized slots and the same slots as the English, "
                  "that also passed an automated meaning check. Spaces between Thai words were removed.",
        downloads=(Download("https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz", "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577", "massive/massive.tar.gz", unpack=True),),
    ),
)
LICENCES = {source.name: source.licence for source in SOURCES}
