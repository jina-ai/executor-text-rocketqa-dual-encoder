import pytest
from docarray import Document, DocumentArray
from executor import RocketQADualEncoder

_EMBEDDING_DIM = 768


@pytest.fixture(scope='session')
def basic_encoder() -> RocketQADualEncoder:
    """
    dual encoder running on GPU
    :return: dual encoder running on GPU
    """
    return RocketQADualEncoder()


def test_no_document(basic_encoder: RocketQADualEncoder):
    """
    none
    :param basic_encoder: encoder
    """
    basic_encoder.encode_passage(None)
    basic_encoder.encode_query(None)


def test_empty_document(basic_encoder: RocketQADualEncoder):
    """
    If the DocumentArray is empty
    :param basic_encoder: encoder
    """
    docs = DocumentArray([])
    basic_encoder.encode_passage(docs)
    assert len(docs) == 0


def test_empty_document_query(basic_encoder: RocketQADualEncoder):
    """
    If the DocumentArray is empty
    :param basic_encoder: encoder
    """
    docs = DocumentArray([])
    basic_encoder.encode_query(docs)
    assert len(docs) == 0


def test_no_text_document(basic_encoder: RocketQADualEncoder):
    """
    If the text property of document is empty
    :param basic_encoder: encoder
    """
    docs = DocumentArray([Document()])
    basic_encoder.encode_passage(docs)
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_no_text_document_query(basic_encoder: RocketQADualEncoder):
    """
    If the text property of document is empty
    :param basic_encoder: encoder
    """
    docs = DocumentArray([Document()])
    basic_encoder.encode_query(docs)
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_encoding_gpu(basic_encoder: RocketQADualEncoder):
    """
    run on CPU, another file is run on GPU
    :param basic_encoder: encoder
    """
    docs = DocumentArray([Document(text="萍水相逢，尽是他乡之客")])
    basic_encoder.encode_passage(docs)
    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)


"""

@pytest.mark.parametrize('batch_size',[1,2,4,8])
def test_batch_size(batch_size:int):
    docs = DocumentArray([Document(text="落霞与孤鹜齐飞，秋水共长天一色") for _ in range(32)])
    encoder_batch = RocketQADualEncoder(batch_size=batch_size)
    encoder_batch.encode_passage(docs)

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)

"""


def test_quality_embeddings(basic_encoder: RocketQADualEncoder):
    """
    Test whether the model can correctly encode the semantic information
    :param basic_encoder: encoder
    """
    docs = DocumentArray(
        [
            Document(id='A', text='渔舟唱晚，响穷彭蠡之滨'),
            Document(id='B', text='雁阵惊寒，声断衡阳之浦'),
            Document(id='C', text='老当益壮，宁移白首之心'),
            Document(id='D', text='穷且益坚，不坠青云之志'),
        ]
    )
    basic_encoder.encode_query(docs)
    docs.match(docs)
    matched = ['B', 'A', 'D', 'C']
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matched[i]
