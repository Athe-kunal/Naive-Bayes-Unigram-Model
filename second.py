from first import LanguageModel
import math
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

if __name__ == "__main__":
    '''
    Iterating over each alpha value
    and returns the log evidence values and test perplexities
    '''
    grid_search_alpha = [int(i) for i in range(1, 11)]

    log_evidences = {}
    test_perplexities = {}
    for alpha in grid_search_alpha:
        print(f"Calculating for alpha {alpha}")
        lm = LanguageModel(alpha)
        log_evidence, train_word_count, word_count, train_length = lm.get_evidence(128)
        log_evidences[alpha] = log_evidence
        pd_dict = lm.get_PD(word_count, train_length)
        _, test_perplexity = lm.compute_perplexity(pd_dict, train_length,train_word_count)
        test_perplexities[alpha] = test_perplexity

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(grid_search_alpha, log_evidences.values(), marker="o", linestyle="dotted")
    ax1.set_ylabel("Log Evidences")

    ax2.plot(
        grid_search_alpha, test_perplexities.values(), marker="o", linestyle="dotted"
    )
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Test Perplexity")

    fig.savefig("Perplexity and Log-evidences.png",bbox_inches='tight')
