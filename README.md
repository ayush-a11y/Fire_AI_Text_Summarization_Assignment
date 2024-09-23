# Fire_AI_Text_Summarization_Assignment
Task is to so text summarization using the simple language model (LLM) models
Models-Bart, T5
evaluation metric- Rogue
Dataset - samsun dataset
The model lis that are perofrm good for text summarization
1. Bert
2. Bart
3. T5
4. Gpt

There mutiple models and also the ditlled model are present but I used the T5 model and bart to compare the performance of models. 
About T5 model
Briefly explaining, T5 (Text-to-Text Transfer Transformer) converts all NLP tasks into a unified "text-to-text" format, meaning that every task—whether it's translation, classification, or summarization—is framed as generating a target sequence from an input sequence. This uniform approach makes T5 highly flexible and efficient for multi-task learning and transfer learning. One key advantage of T5 is that it has different model sizes (from T5-small to T5-xxl), allowing for more scalable performance depending on the computational resources available.
About Bart model
 BART is a denoising autoencoder that employs the strategy of distorting the input text in many ways, such as blanking out some words and flipping them around, and then learning to reconstruct it. BART has outperformed established models like RoBERTa and BERT on multiple NLP benchmarks, and it is especially efficient in summarization tasks, due to its ability to generate text and learn the context of the input text.

 Evaluating performance for language models can be quite tricky, especially when it comes to text summarization. The goal of our model is to produce a short sentence describing the content of a dialogue, while maintaining all the important information within that dialogue.

One of the quantitative metrics we can employ to evaluate performance is the ROUGE Score. It is considered one of the best metrics for text summarization and it evaluates performance by comparing the quality of a machine-generated summary to a human-generated summary used for reference.

The similarities between both summaries are measured by analyzing the overlapping n-grams, either single words or sequences of words that are present in both summaries. These can be unigrams (ROUGE-1), where only the overlap of sole words is measured; bigrams (ROUGE-2), where we measure the overlap of two-word sequences; trigrams (ROUGE-3), where we measure the overlap of three-word sequences; etc. Besides that, we also have:

• ROUGE-L: It measures the Longest Common Subsequence (LCS) between the two summaries, which helps to capture content coverage of the machine-generated text. If both summaries have the sequence "the apple is green", we have a match regardless of where they appear in both texts.

• ROUGE-S: It evaluates the overlap of skip-bigrams, which are bigrams that permit gaps between words. This helps to measure the coherence of a machine-generated summary. For example, in the phrase "this apple is absolutely green", we find a match for the terms such as "apple" and "green", if that is what we are looking for.

Result
Bart got much higher accuracy from the t5 model 

Conlusion
Bart is more efficent in giving the corretc summarization

Resource constraint
I want to train the facebook/bart-large-xsum' but i ran this both on colab there is some ram error and for T5 large there is gpu exhuast why i ran it on colab becaus ein local laptop ther is no gpu so it will run fatset but there are some constrainst of gpu usage of google colab so that is why i trained the t5-samll and baart-base 

