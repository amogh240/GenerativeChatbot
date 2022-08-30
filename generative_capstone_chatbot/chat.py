import numpy as np
import re
from test_model import training_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features, \
    target_features, reverse_target_features, max_decoder_seq_length, max_encoder_seq_length, encoder_model
from collections import Counter
from nltk import pos_tag
from functions import compare_overlap, compute_similarity, extract_nouns, preprocess
from food_responses import food_responses
from sports_responses import sport_responses


class ChatBot:
    negative_commands = ["stop", "no", "nothing", "you suck"]
    exit_commands = ["exit", "stop", "done"]
    food_words = ["food", "appetizer", "appetizer", "dessert", "dessert", "meat"]
    sports_words = ["sports", "game", "sport", 'games', "ball", "balls"]

    def start_chat(self):
        reply = input("Hi, my name is Chatty the Chatbot. Would you like to chat with me today?")
        if reply in self.negative_commands:
            print("Okay, have a great day!")
            return
        self.chat(reply)

    def get_exit(self, reply):
        for i in self.exit_commands:
            if i in reply:
                print("Have a great day!")
                return True
        return False

    def chat(self, reply):
        while not self.get_exit(reply):
            for word in self.food_words:
                if word in reply:
                    reply = input(self.find_food_intent_match(reply))
                    break
            else:
                for sport_word in self.sports_words:
                    if sport_word in reply:
                        reply = input(self.find_sport_intent_match(reply))
                else:
                    reply = input(self.gen_response(reply))

    def string_to_matrix(self, reply):
        tokens = re.findall(r"[\w']+|[^\s\w]", reply)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')

        for timestep, token in enumerate(tokens):
            if token in input_features.keys():
                user_input_matrix[0, timestep, input_features[token]] = 1.
        return user_input_matrix

    def gen_response(self, reply):
        reply_matrix = self.string_to_matrix(reply)
        states_value = encoder_model.predict(reply_matrix)
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        decoded_sentence = ''
        stop = False
        while not stop:
            output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features[sampled_token_index]
            decoded_sentence += " " + sampled_token

            target_seq = np.zeros((1, 1, num_decoder_tokens))

            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [hidden_state, cell_state]

        return decoded_sentence

    def find_food_intent_match(self, reply):
        bow_user_message = Counter(preprocess(reply))
        bow_responses = [Counter(preprocess(response)) for response in food_responses]
        similarity_list = [compare_overlap(response, reply) for response in bow_responses]
        response_index = similarity_list.index(max(similarity_list))
        return food_responses[response_index]

    def find_sport_intent_match(self, reply):
        bow_user_message = Counter(preprocess(reply))
        bow_responses = [Counter(preprocess(response)) for response in sport_responses]
        similarity_list = [compare_overlap(response, bow_user_message) for response in bow_responses]
        response_index = similarity_list.index(max(similarity_list))
        return sport_responses[response_index]


chatter = ChatBot()
chatter.start_chat()
