import math
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
plt.style.use("ggplot")


class LanguageModel:
    '''
    LanguageModel implements an unigram model and calculates the
    MLE, MAP and PD estimates

    Attributes:
        alpha: float = Dirichlet distribution hyperparameter
        train_file: str = path of training file
        test_file: str = path of testing file
    '''
    def __init__(
        self,
        alpha=2.0,
        train_file="pp1data\training_data.txt",
        test_file="pp1data\test_data.txt",
    ):
        self.alpha = alpha
        self.train_vocab_list, self.train_len = self.process_data(train_file)
        self.test_vocab_list, self.test_len = self.process_data(test_file)
        self.unique_words_list = self.get_unique_words()
        #K is the length of number of unique words in train and test
        self.K = len(self.unique_words_list)
        self.test_word_count = self.get_test_word_count()

    def read_data(self, file_name):
        file_name = file_name.split("\t")
        file_name.insert(1, "\\t")
        file_name = "".join(file_name)
        with open(file_name, "r") as f:
            lines = f.readlines()

        return lines

    def process_data(self, file_name):
        lines = self.read_data(file_name)
        vocab_list = lines[0].split(" ")[:-1]
        vocab_length = len(vocab_list)

        return vocab_list, vocab_length

    def get_unique_words(self):
        '''
        Returns the list of unique words
        '''
        train_set = set(self.train_vocab_list)
        test_set = set(self.test_vocab_list)
        assert (
            train_set == test_set
        ), "The train and test set must have same unique words"
        train_test_list = set(list(train_set)+list(test_set))
        return list(train_test_list)

    def get_test_word_count(self):
        test_word_count = {word: 0 for word in self.unique_words_list}
        for test_word in self.test_vocab_list:
            test_word_count[test_word] += 1
        return test_word_count

    def get_frequency(self, length):
        '''
        Returns the frequency of words in unique words list and 
        training list.
        '''
        train_length = int(self.train_len / length)
        req_vocab_list = self.train_vocab_list[:train_length]

        train_word_count_list = []
        train_word_count = {i:0 for i in req_vocab_list}
        all_unique_word_count = {i: 0 for i in self.unique_words_list}

        for word in req_vocab_list:
            train_word_count_list.append(word)
            train_word_count[word]+=1
            all_unique_word_count[word] += 1

        return all_unique_word_count, train_length, train_word_count_list,train_word_count

    def get_MLE(self, word_count: dict, train_length: int):
        '''
        Computes the probability of frequency 
        of each word in train set
        '''
        likelihood = {word: f / train_length for word, f in word_count.items()}
        return likelihood

    def get_MAP(self, word_count: dict, train_length: int):
        '''
        Computes the MAP estimate
        '''
        alpha_not = self.alpha * self.K

        map_dict = {}
        for word, freq in word_count.items():
            map_dict[word] = (freq + self.alpha - 1.0) / (
                train_length + alpha_not - self.K
            )

        return map_dict

    def get_PD(self, word_count: dict, train_length: int):
        '''
        Computes the PD estimate
        '''
        alpha_not = self.alpha * self.K

        pd_dict = {}
        for word, freq in word_count.items():
            pd_dict[word] = (freq + self.alpha) / (train_length + alpha_not)

        return pd_dict

    def get_evidence(self, length):
        '''
        Computes the log evidence values
        for the training set 
        '''
        word_count, train_length, train_words_list,train_word_count = self.get_frequency(length)
        num = 0
        denom = 0
        for word, freq in train_word_count.items():
            num += math.log(math.factorial(self.alpha + freq - 1))
            denom += math.log(math.factorial(self.alpha - 1))
        alpha_not = self.K * self.alpha
        num += math.log(math.factorial(alpha_not - 1))
        denom += math.log(math.factorial(alpha_not + train_length - 1))
        log_evidence = num - denom
        return log_evidence, train_word_count, word_count, train_length

    def get_estimates(self, length):
        word_count, train_length, train_words_list,train_word_count = self.get_frequency(length)
        likelihood_estimate = self.get_MLE(word_count, train_length)
        map_estimate = self.get_MAP(word_count, train_length)
        pd_estimate = self.get_PD(word_count, train_length)

        return (
            likelihood_estimate,
            map_estimate,
            pd_estimate,
            train_length,
            train_words_list,
            train_word_count
        )

    def get_perplexities(self, length):
        (
            likelihood_estimate,
            map_estimate,
            pd_estimate,
            train_length,
            train_words_list,
            train_word_count
        ) = self.get_estimates(length)
        train_mle_perplexity, test_mle_perplexity = self.compute_perplexity(
            likelihood_estimate, train_length,train_word_count, train_words_list
        )
        train_map_perplexity, test_map_perplexity = self.compute_perplexity(
            map_estimate, train_length,train_word_count, train_words_list
        )
        train_pd_perplexity, test_pd_perplexity = self.compute_perplexity(
            pd_estimate, train_length,train_word_count, train_words_list
        )

        return (
            train_mle_perplexity,
            train_map_perplexity,
            train_pd_perplexity,
            test_mle_perplexity,
            test_map_perplexity,
            test_pd_perplexity,
        )

    def compute_perplexity(
        self, compute_dict: dict, train_length: int, train_word_count:dict,
        train_words_list: list = []
    ):
        '''
        Returns the perplexity values of train and test set
        given the training length
        '''
        train_log_vals = 0
        test_log_vals = 0
        train_words = list(set(train_words_list))
        train_compute_dict = {word: compute_dict[word] for word in train_words}

        for keys, val in train_compute_dict.items():
            train_log_vals += train_word_count[keys]*math.log(val)

        for keys, val in compute_dict.items():
            if val == 0:
                test_log_vals += -math.inf
            else:
                test_log_vals += self.test_word_count[keys]*math.log(val)

        train_perplexity = math.exp(-train_log_vals / train_length)
        test_perplexity = math.exp(-test_log_vals / self.test_len)

        return train_perplexity, test_perplexity

    def run(self, lengths: list):
        train_results_dict = {}
        test_results_dict = {}
        for length in lengths:
            train_key_val = f"N/{length} [TRAIN]"
            test_key_val = f"N/{length} [TEST]"
            (
                train_mle_perplexity,
                train_map_perplexity,
                train_pd_perplexity,
                test_mle_perplexity,
                test_map_perplexity,
                test_pd_perplexity,
            ) = self.get_perplexities(length)
            train_results_dict[train_key_val] = {
                "MLE": train_mle_perplexity,
                "MAP": train_map_perplexity,
                "PD": train_pd_perplexity,
            }
            test_results_dict[test_key_val] = {
                "MLE": test_mle_perplexity,
                "MAP": test_map_perplexity,
                "PD": test_pd_perplexity,
            }

        return train_results_dict, test_results_dict

    def plot_bars(self, results_dict: dict):
        '''
        Plots the bar charts
        '''
        title = list(results_dict.keys())[0].split(" ")[1][1:-1]

        x = list(results_dict.keys())
        x = [keys.split(" ")[0] for keys in x]

        y = [val.values() for val in results_dict.values()]
        y1 = [list(v)[0] for v in y]
        y2 = [list(v)[1] for v in y]
        y3 = [list(v)[2] for v in y]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 0.3
        ind = [0, 1, 2, 3, 4]

        rects1 = ax.bar(
            [i + width * 1 for i in ind], y1, width, color="r", align="center"
        )
        rects2 = ax.bar(
            [i + width * 2 for i in ind], y2, width, color="b", align="center"
        )
        rects3 = ax.bar(
            [i + width * 3 for i in ind], y3, width, color="g", align="center"
        )

        # ax.set_ylim(1.0, 1.00175)
        ax.set_ylabel("Perplexity")
        ax.set_xticks([i + width for i in ind])
        ax.set_xticklabels((x[0], x[1], x[2], x[3], x[4]))
        ax.legend((rects1[0], rects2[0], rects3[0]), ("MLE", "MAP", "PD"))
        ax.set_title(f"PLOT FOR {title} SET")
        plt.savefig(f"{title}.jpg",bbox_inches='tight')
        plt.show()

    def plot_scatter(self, results_dict: dict, saveName: str = ""):
        title = list(results_dict.keys())[0].split(" ")[1][1:-1]

        X = [self.train_len // i for i in [128, 64, 16, 4, 1]]
        y = [val.values() for val in results_dict.values()]
        y1 = [list(v)[0] for v in y]
        y2 = [list(v)[1] for v in y]
        y3 = [list(v)[2] for v in y]

        max_val = max(y2) + 2000
        y1 = [max_val if i == math.inf else i for i in y1]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(X, y1, marker="o", label="MLE", linestyle="dotted")
        ax.plot(X, y2, marker="s", label="MAP", linestyle="dotted")
        ax.plot(X, y3, marker="o", label="PD", linestyle="dotted")

        ax.legend()
        ax.set_title(f"PLOT FOR {title} SET")
        ax.set_xlabel("Train set size")
        ax.set_ylabel("Perplexity")
        fig.savefig(f"{title}{saveName}.png",bbox_inches='tight')
        print(f"The figure is saved as {title}{saveName}.png")
        plt.show()

    def combined_plot(self,train_results_dict:dict,test_results_dict:dict):

        title = "COMBINED PLOT"
        X = [self.train_len // i for i in [128, 64, 16, 4, 1]]
        y_train = [val.values() for val in train_results_dict.values()]
        y1 = [list(v)[0] for v in y_train]
        y2 = [list(v)[1] for v in y_train]
        y3 = [list(v)[2] for v in y_train]

        y_test = [val.values() for val in test_results_dict.values()]
        y4 = [list(v)[0] for v in y_test]
        y5 = [list(v)[1] for v in y_test]
        y6 = [list(v)[2] for v in y_test]
        max_val = max(y5) + 2000
        y4 = [max_val if i == math.inf else i for i in y4]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(1,13000)
        ax.plot(X, y1, marker="o", label="TRAIN MLE", linestyle="dotted")
        ax.plot(X, y2, marker="s", label="TRAIN MAP", linestyle="dotted")
        ax.plot(X, y3, marker="o", label="TRAIN PD", linestyle="dotted")
        ax.plot(X, y4, marker="o", label="TEST MLE", linestyle="dotted")
        ax.plot(X, y5, marker="o", label="TEST MAP", linestyle="dotted")
        ax.plot(X, y6, marker="o", label="TEST PD", linestyle="dotted")
        ax.set_title("COMBINED PLOT")
        ax.legend()
        ax.set_xlabel("Train set size")
        ax.set_ylabel("Perplexity")
        fig.savefig("Combined Plot.png",bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    lm = LanguageModel()
    train_results_dict, test_results_dict = lm.run([128, 64, 16, 4, 1])
    print(train_results_dict)
    print("#"*50)
    print(test_results_dict)
    lm.plot_scatter(train_results_dict)
    lm.plot_scatter(test_results_dict)
    lm.combined_plot(train_results_dict,test_results_dict)
