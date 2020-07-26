import numpy as np
import math
import time
import argparse

parser = argparse.ArgumentParser(
    description="Per generate embedding weights"
)

parser.add_argument('--embedding-size', type=str,
                    default='39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--output', type=str)
args = parser.parse_args()

embedding_size = np.fromstring(args.embedding_size, dtype=int, sep="-")
total_embedding_size = sum(embedding_size)

print('Embedding size: ', embedding_size)
print('Write model file to ', args.output)

with open(args.output, 'wb') as f:
    m = 0
    for i, n in enumerate(embedding_size):
        rest = n
        while rest > 0:
            s = min(rest, 2048)
            c1 = np.arange(start=m, stop=m + s)
            c2 = np.repeat(a=i, repeats=s)
            c3 = np.random.uniform(
                low=-math.sqrt(1 / n), high=math.sqrt(1 / n), size=(s, args.dim)).astype(np.float32)
            u1 = np.frombuffer(c1.tobytes(), dtype=np.uint8).reshape((s, 8))
            u2 = np.frombuffer(c2.tobytes(), dtype=np.uint8).reshape((s, 8))
            u3 = np.frombuffer(c3.tobytes(), dtype=np.uint8).reshape(
                (s, 4 * args.dim))
            u = np.concatenate((u1, u2, u3), axis=1)
            f.write(u.tobytes())
            m += s
            rest -= s
            print('Writing {:.2f}% for column {}({}) and {:.2f}% in total'.format(
                (n - rest) / n * 100, i, n, m / total_embedding_size * 100), end='\r')

        print()
