import rocketqa
from jina import Executor, requests
from docarray import DocumentArray
from jina.logging.logger import JinaLogger


class RocketQADualEncoder(Executor):
    """
    rocketQAdualEncoder
    """

    def __init__(
        self,
        model_name="zh_dureader_de",
        use_cuda=True,
        device_id=0,
        batch_size=32,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.available_models_list = list(rocketqa.available_models())
        """
        available_models_list = ['v1_marco_de', 'v1_marco_ce', 'v1_nq_de',
                            'v1_nq_ce', 'pair_marco_de', 'pair_nq_de',
                            'v2_marco_de', 'v2_marco_ce', 'v2_nq_de',
                            'zh_dureader_de', 'zh_dureader_ce',
                            'zh_dureader_de_v2', 'zh_dureader_ce_v2']
        """
        if '_de' not in model_name:
            raise ValueError('need de model name')
        if model_name not in self.available_models_list:
            raise ValueError(
                f'The ``model_name`` parameter should be in available models list, but got {model_name}'
            )
        self.model = rocketqa.load_model(
            model=model_name,
            use_cuda=use_cuda,
            device_id=device_id,
            batch_size=batch_size,
        )

    @requests(on="/index")
    def encode_passage(self, docs: DocumentArray, **kwargs):
        """
        Encode the document to be queried
        :param docs: documents sent to the encoder.
        :param **kwargs: parameter for keyword arguments.
        """
        if docs is not None:
            if len(docs) != 0:
                c_docs = DocumentArray(filter(lambda x: bool(x.text), docs))
                if len(c_docs) != 0:
                    c_docs.embeddings = list(self.model.encode_para(para=c_docs.texts))

    @requests(on="/search")
    def encode_query(self, docs: DocumentArray, **kwargs):
        """
        Encode the question
        :param docs: documents sent to the encoder.
        :param **kwargs: parameter for keyword arguments.
        """
        if docs is not None:
            if len(docs) != 0:
                c_docs = DocumentArray(filter(lambda x: bool(x.text), docs))
                if len(c_docs) != 0:
                    c_docs.embeddings = list(
                        self.model.encode_query(query=c_docs.texts)
                    )
