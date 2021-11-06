import tensorflow as tf
from transformers import BertTokenizer
import tensorflow_hub as hub

save_path = '/bert-qa-saved-tf-model/1'
model = tf.saved_model.load(save_path)
print('loaded saved model')

model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_model_response(user_question_dict):
  print('function called')

  question = user_question_dict.get('question')
  paragraph = '''A little old woman baked a gingerbread man and when she took him out of the oven, 
    he ran away. The woman and her husband chased him, as well as the pig, cow and horse. No one 
    could catch him. He came to a river and a sly fox told him he could jump on his tail and he 
    would take him across. He did and the fox went deeper and the gingerbread man had to jump on 
    his back and then on his nose. When he got to his nose, the fox ate him.'''

  # create tokens of questions and input paragraph
  question_tokens = tokenizer.tokenize(question)
  paragraph_tokens = tokenizer.tokenize(paragraph)
  tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']

  # create input word ids
  input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_word_ids)
  input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)
  input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
      tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))

  outputs = model([input_word_ids, input_mask, input_type_ids])

  # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
  short_start = tf.argmax(outputs[0][0][1:]) + 1
  short_end = tf.argmax(outputs[1][0][1:]) + 1
  answer_tokens = tokens[short_start: short_end + 1]
  answer = tokenizer.convert_tokens_to_string(answer_tokens)

  answer = {'answer': answer}
  return answer
