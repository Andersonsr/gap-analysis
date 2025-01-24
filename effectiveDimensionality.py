import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def effective_dimension(data, threshold, accumulate=False):
    cov_m = np.cov(data.transpose())
    U, S, Vh = np.linalg.svd(cov_m, full_matrices=True)

    if accumulate:
        sum = np.sum(S, axis=0)
        accumulated_significance = [np.sum(S[:i])/sum for i in range(len(S))]
        return accumulated_significance
    else:
        return np.log(S)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', '-d', type=str, help='.pkl file to load ', required=True)
    parser.add_argument('--vlm', '-v', type=str, help='vlm name to use in title', required=True)
    parser.add_argument('--accumulate', '-a', action='store_true', help='accumulated singular values proportion',
                        default=False)
    parser.add_argument('--threshold', '-t', type=float, help='threshold of the accumulated singular values',
                        default=0.99)
    args = parser.parse_args()
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
        image_embeddings = [d.squeeze(0).cpu().detach().numpy() for d in data['image_embeddings']]
        image_embeddings = np.array(image_embeddings)
        image_significance = effective_dimension(image_embeddings, args.threshold, accumulate=args.accumulate)

        text_embeddings = [d[0].cpu().detach().numpy() for d in data['texts_embeddings']]
        text_embeddings = np.array(text_embeddings)
        text_significance = effective_dimension(text_embeddings, args.threshold, accumulate=args.accumulate)

        plt.plot(range(image_embeddings.shape[1]), image_significance, label='images coco test')
        plt.plot(range(image_embeddings.shape[1]), text_significance, label='texts coco test')

        plt.title(f"Effective dimension analysis {args.vlm}")
        plt.xlabel("dimension")
        if args.accumulate:
            plt.ylabel("Accumulated significance")
            for i, e in enumerate(image_significance):
                if e >= args.threshold:
                    plt.plot(i, e, color='red', marker='x', label=f'threshold {args.threshold}')
                    plt.annotate(str(i), xy=(i, e))
                    # print(i, e)
                    break

            for i, e in enumerate(text_significance):
                if e >= args.threshold:
                    plt.plot(i, e, color='red', marker='x')
                    plt.annotate(str(i), xy=(i, e))
                    # print(i, e)
                    break

        else:
            plt.ylabel("log of singular values")
        plt.legend()
        plt.show()

