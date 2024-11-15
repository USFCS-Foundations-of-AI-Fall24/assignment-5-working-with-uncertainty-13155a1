

import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

# hidden states들을 찾아내라.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""
        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    # lander.emit, lander.trans에서 init에 나와 있는 것과 같은 데이터를 불러오는 함수
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        trans_file = f"{basename}.trans"
        if os.path.exists(trans_file):
            with open(trans_file, "r") as file:
                for line in file:
                    state1, state2, prob = line.split()
                    prob = float(prob)
                    if state1 not in self.transitions:
                        self.transitions[state1] = {}
                    self.transitions[state1][state2] = prob

        # Load emissions
        emit_file = f"{basename}.emit"
        if os.path.exists(emit_file):
            with open(emit_file, "r") as file:
                for line in file:
                    state, output, prob = line.split()
                    prob = float(prob)
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][output] = prob


    ## you do this.
    # 5-6 line code 정도일 것임.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        states = ['#']
        emissions = []
        
        for _ in range(n):
            # add next state
            cur_state = states[-1]
            next_state = random.choices(
                population=list(self.transitions[cur_state].keys()),
                weights=list(map(float, self.transitions[cur_state].values()))
            )[0]
            states.append(next_state)
            # print("next_state: ", next_state)

            # add next emission
            emission = random.choices(
                population=list(self.emissions[next_state].keys()),
                weights=list(map(float, self.emissions[next_state].values()))
            )[0]
            emissions.append(emission)

        return Sequence(states[1:], emissions)

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi(forward?) algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

    # 달라지는 two big things
    # max instead of sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Simulation")
    parser.add_argument("basename", type=str, help="basename")
    parser.add_argument("--generate", type=int, help="method - generate")
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)
    # print("transitions: ", hmm.transitions)
    # print("emissions: ", hmm.emissions)

    if args.generate:
        sequence = hmm.generate(args.generate)
        print(" ".join(sequence.stateseq)) 
        print(" ".join(sequence.outputseq))

