import math
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
plt.style.use("ggplot")


class AuthorIdentificationModel:
    def __init__(
        self,
        alpha=2.0,
    ):
        '''
        AuthorIdentificationModel implements an unigram model and calculates the
        perplexity difference between writings of different authors

        Attributes:
            alpha: float = Dirichlet distribution hyperparameter
        '''

        self.alpha = alpha
        self.train_data = self.read_data("pp1data\pg121.txt.clean")
        self.train_len = len(self.train_data)
        self.test_data1 = self.read_data("pp1data\pg141.txt.clean")
        self.test1_len = len(self.test_data1)
        self.test_data2 = self.read_data("pp1data\pg1400.txt.clean")
        self.test2_len = len(self.test_data2)

        self.unique_words_list = self.get_unique_words()
        self.K = len(self.unique_words_list)

        self.test1_word_count, self.test2_word_count = self.get_test_word_count()

    def read_data(self, file_name):
        '''
        Reading and preprocessing function
        '''
        with open(file_name, "r") as f:
            lines = [line for line in f.readlines() if line.strip()]
        l = [line.split("\n") for line in lines]
        all_words = []
        for k in l:
            for j in k[0].split(" "):
                if j == "" or j == " ":
                    continue
                else:
                    all_words.append(j)
        return all_words[:-1]

    def get_unique_words(self):
        combined_list = self.train_data + self.test_data1 + self.test_data2
        unique_words_list = list(set(combined_list))
        return unique_words_list

    def get_test_word_count(self):
        test_word_count1 = {word: 0 for word in self.unique_words_list}
        test_word_count2 = {word: 0 for word in self.unique_words_list}
        for test_word in self.test_data1:
            test_word_count1[test_word] += 1
        for test_word in self.test_data2:
            test_word_count2[test_word] += 1
        return test_word_count1, test_word_count2

    def get_frequency(self):
        '''
        Returns the word count of unqiue words list
        '''
        word_count = {i: 0 for i in self.unique_words_list}

        for word in self.train_data:
            word_count[word] += 1

        assert (
            len(word_count) == self.K  # 17329
        ), "The length of the word count and the unique word counts should be the same"

        return word_count

    def get_PD(self, word_count: dict):
        '''
        Computes the Predictive distribution
        '''
        alpha_not = self.alpha * self.K

        pd_dict = {}
        for word, freq in word_count.items():
            pd_dict[word] = (freq + self.alpha) / (self.train_len + alpha_not)

        return pd_dict

    def compute_test_perplexity(self, compute_dict: dict):
        '''
        Computes the test perplexity on pg141 (test1_log_vals)
        and pg1400 (test2_log_vals)
        '''
        test1_log_vals = 0
        test2_log_vals = 0

        for keys, val in compute_dict.items():
            if val == (0):
                test1_log_vals += -math.inf
                test2_log_vals += -math.inf
            else:
                test1_log_vals += self.test1_word_count[keys] * math.log(val)
                test2_log_vals += self.test2_word_count[keys] * math.log(val)

        test1_perplexity = math.exp(-test1_log_vals / len(self.test_data1))
        test2_perplexity = math.exp(-test2_log_vals / len(self.test_data2))

        return test1_perplexity, test2_perplexity

    def run(self):
        '''
        Main function to get results and plot them
        '''
        word_count = self.get_frequency()
        pd_dict = self.get_PD(word_count)

        test1_perplexity, test2_perplexity = self.compute_test_perplexity(pd_dict)

        return_dict = {"test1:pg141": test1_perplexity, "test2:pg1400": test2_perplexity}

        fig = plt.figure()
        ax = fig.add_subplot(111)
        y = list(return_dict.values())
        x = list(return_dict.keys())
        ax.plot(x,y,marker="o",linestyle='None')

        ax.annotate(str(round(y[0],2)),xy=(x[0],y[0]))
        ax.annotate(str(round(y[1],2)),xy=(x[1],y[1]))
        ax.set_ylabel("Perplexity")
        ax.set_xlabel("Test Document")

        fig.savefig("Author Idenitification.png")
        plt.show()

        return return_dict


if __name__ == "__main__":
    authorModel = AuthorIdentificationModel()
    print(authorModel.run())
