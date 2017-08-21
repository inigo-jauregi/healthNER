# healthNER

This is the code used for the research in our paper "_Recurrent neural networks with specialized word embeddings for health-domain named-entity recognition_".
It is a little scattered, but fully functioning.

There code contains three main files:

1.__CRF__: This is the file where we implement the Conditional Random Field (CRF) model which in the paper's results section has the same name (Do not confuse it with the prediction layer of the B-LSTM-CRF). It contains two main files. The HCRF2.0b open-source tool to train a CRF model and a set of files for data_preparation. Further description in the [CRF] file.

2.__Bidirectional_LSTM-CRF__: This is the file where we implement the Bidirectional LSTM and the Bidirectional-LSTM-CRF models which in the paper's result section have the same name. The code is created following the code of [Bidirectional-LSTM-CRF-for-Clinical-Concept-Extraction] created by Raghav Chalapathy. Further description in the [Bidirectional_LSTM-CRF] file.

3.__Bidirectional_LSTM-CRF_plus_feature_engineering__: This is identical to the [Bidirectional_LSTM-CRF] with the only extra option of addind the hand-crafted features described in the paper. Further description in the [Bidirectional_LSTM-CRF_plus_feature_engineering] file.

*Due to the large size of the word embedding files, they have not been provided here. CommonCrawl embeddings are available in the oficial GloVe webpage. In order to obtain the specialized embeddings used in the paper, you can contact: _ijauregi@cmcrc.com_.

__NOTE:__ The commit number of the code used in the paper is e5dc2a0. 


[CRF]: https://github.com/ijauregiCMCRC/healthNER/tree/master/CRF
[Bidirectional-LSTM-CRF-for-Clinical-Concept-Extraction]: https://github.com/raghavchalapathy/Bidirectional-LSTM-CRF-for-Clinical-Concept-Extraction
[Bidirectional_LSTM-CRF]: https://github.com/ijauregiCMCRC/healthNER/tree/master/Bidirectional_LSTM-CRF
[Bidirectional_LSTM-CRF_plus_feature_engineering]: https://github.com/ijauregiCMCRC/healthNER/tree/master/Bidirectional_LSTM-CRF_plus_feature_engineering
