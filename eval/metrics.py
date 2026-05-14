from rouge_score import rouge_scorer
import sacrebleu

_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(hyp, ref):
    if not hyp or not ref:
        return 0.0
    return _sc.score(ref, hyp)["rougeL"].fmeasure


def bleu4(hyp, ref):
    if not hyp or not ref:
        return 0.0
    return sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0


if __name__ == "__main__":
    a = "the cat sat on the mat"
    b = "the cat sat on the mat"
    c = "a dog ran through the park"

    print("identical:", rouge_l(a, b), bleu4(a, b))
    print("different:", rouge_l(a, c), bleu4(a, c))
    print("empty hyp:", rouge_l("", b), bleu4("", b))
