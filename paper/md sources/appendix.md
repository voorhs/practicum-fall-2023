## Pairwise Model

Как упоминалось в разделе \ref{sec:aug}, мы использовали модель для измерения близости между репликами в диалоге. В базовом варианте её можно было бы реализовать так: взять сентенс эмбеддинги реплик и сравнивать косинусы между ними. Обучение такой модели на последовательных репликах с помощью контрастивного обучения дает неплохие уттеранс эмбеддинги (цитата amazon dse).

У этого способа один существенный недостаток: он не учитывает контекст из нескольких последних реплик. В случае диалога крайне важно сравнивать не просто пару реплик, а пару контекст—ответ. Поэтому мы использовали следующую модель (ссыль на картинку):

- сначала все уттерансы контекста $c=[u_1,\ldots,u_k]$ и ответа $r$ кодируются с помощью одного сентенс эмбеддера
- эмбеддинги контекста конкатенируются и подаются на вход проектору, который выдает вектор, репрезентующий контекст
- эмбеддинг ответа подается на вход второго проектора и получается вектор, репрезентующий ответ

- между полученными векторами считается косинус как мера близости контекста и ответа

В качестве сентенс эмбеддера был использован aws-ai/dse-bert-large. Модель обучена контрастивным лоссом с in-batch negative sampling со следующими параметрами

- batch size 128
- temperature 0.05
- fine tune layers 3
- context size 3
- projection size 256

Батч формировался из пар “контекст-ответ” со всего диалогового датасета, поэтому негативными примерами служили не семплы из этого же диалога, а совершенно случайные со всего датасета. Это позволяет коллектить батч произвольного размера, а не размера, соответственного диалогу, что делает претрейн задачу более сложной.

Полученная модель сильно напоминает модель получения уттеранс эмбеддингов ConveRT. Недостатком последней является во-первых то, что она закрытая, во-вторых, ее архитектура слишком специфична и не использует всем привычный BERT-like бекбон.

В итоге получилась такая модель, которая способна распознавать отдельные стейджи в диалоге (невсегда):

(картинка с хитмапом)

## Pairwise Model

As mentioned in Section \ref{sec:aug}, we utilized a model for measuring the similarity between dialog utterances. In its basic form, it can be implemented as follows: take sentence embeddings of the utterances and compare the cosine similarities between them. Training such a model on sequential utterances using contrastive learning yields commendable utterance embeddings (as quoted by Amazon DSE).

However, this approach has a significant drawback; it does not consider the context from several preceding utterances. In a dialogue, it is crucial to compare not just pairs of utterances but pairs of context-response. Therefore, we employed the following model (referenced in the figure):

- First, embeddings for all context utterances $c=[u_1,\ldots,u_k]$ and the response $r$ are obtained using a pretrained sentence encoder.
- The embeddings of the context are concatenated and passed to a projector that outputs a vector representing the context.
- The response embedding is fed into a second encoder, resulting in a vector representing the response.

- The cosine similarity between the obtained vectors is computed as a measure of the context and response similarity.

The `aws-ai/dse-bert-large` model was used as the sentence encoder. The model was trained with a contrastive loss using in-batch negative sampling, with the following parameters:

- Batch size: 128
- Temperature: 0.05
- Fine-tune layers: 3
- Context size: 3
- Projection size: 256

Batches were formed from "context-response" pairs from the entire dialog dataset, where negative examples were not samples from the same dialog but entirely random examples from the dataset. This allows batching of arbitrary sizes, not limited to the dialog size, making the pre-training task more challenging.

The resulting model closely resembles the ConveRT model for obtaining utterance embeddings. The drawbacks of the latter model are, firstly, that it is proprietary, and secondly, its architecture is highly specific and does not utilize the familiar BERT-like backbone.

As a result, we obtained a model capable of recognizing individual stages in a dialogue (not always):

(heatmap image)