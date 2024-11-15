

import random
import argparse
import codecs
import os
import numpy as np

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

    def forward(self, sequence, basename):
        states = list(self.transitions.keys())
        states.remove("#")

        N = len(states) 
        T = len(sequence) 

        alpha = np.zeros((T, N))

        first_obs = sequence[0]
        
        start_prob = 1 / N
        for s in range(N):
            state = states[s]
            alpha[0, s] = start_prob * self.emissions[state].get(first_obs, 0)

        for t in range(1, T):
            obs = sequence[t]
            for s in range(N):
                state = states[s]
                trans_probs = [
                    alpha[t - 1, prev_s] * self.transitions[states[prev_s]].get(state, 0)
                    for prev_s in range(N)
                ]
                alpha[t, s] = max(trans_probs) * self.emissions[state].get(obs, 0)

        best_last_state_index = np.argmax(alpha[T - 1, :])
        best_last_state = states[best_last_state_index]

        if basename == "lander":
            safe_landings = {"2,5", "3,4", "4,3", "4,4", "5,5"}
            is_safe = "Yes" if best_last_state in safe_landings else "No"
            return best_last_state, is_safe
        
        return best_last_state

    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

    # 달라지는 two big things
    # max instead of sum
        EPSILON = 1e-10 

        states = list(self.transitions.keys())
        states.remove("#") 

        N = len(states)
        T = len(sequence) 

        viterbi_matrix = np.zeros((T, N))
        backpointer_matrix = np.zeros((T, N), dtype=int) 

        first_obs = sequence[0]
        for s in range(N):
            state = states[s]
            viterbi_matrix[0, s] = (1 / N) * self.emissions[state].get(first_obs, EPSILON)
            backpointer_matrix[0, s] = 0 

        for t in range(1, T):
            obs = sequence[t]
            for s in range(N):
                state = states[s]
                trans_probs = [
                    viterbi_matrix[t - 1, prev_s] * self.transitions[states[prev_s]].get(state, EPSILON)
                    for prev_s in range(N)
                ]
                max_prob = max(trans_probs)
                viterbi_matrix[t, s] = max_prob * self.emissions[state].get(obs, EPSILON)
                backpointer_matrix[t, s] = np.argmax(trans_probs)

        best_last_state_index = np.argmax(viterbi_matrix[T - 1, :])
        best_last_state = states[best_last_state_index]

        best_path = [best_last_state]
        for t in range(T - 1, 0, -1):
            best_last_state_index = backpointer_matrix[t, best_last_state_index]
            best_path.insert(0, states[best_last_state_index])

        return best_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Simulation")
    parser.add_argument("basename", type=str, help="basename")
    parser.add_argument("--generate", type=int, help="method - generate")
    parser.add_argument("--forward", type=str, help="method - forward")
    parser.add_argument("--viterbi", type=str, help="method - viterbi")
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)
    # print("transitions: ", hmm.transitions)
    # print("emissions: ", hmm.emissions)

    if args.generate:
        sequence = hmm.generate(args.generate)
        print(" ".join(sequence.stateseq)) 
        print(" ".join(sequence.outputseq))
        with open(f"{args.basename}_sequence.obs", "w") as f:
            f.write("\n" + ' '.join(sequence.outputseq) + "\n")
    
    if args.forward:
        with open(args.forward, 'r') as file:
            sequence = file.read().strip().split()
        
        if args.basename == "lander" :
            best_states, is_safe = hmm.forward(sequence, args.basename)
            print("Most likely hidden states: ", best_states)
            print("Safe: ", is_safe)
        else:
            best_states = hmm.forward(sequence, args.basename)
            print("Most likely hidden states: ", best_states)

    if args.viterbi:
        with open(args.viterbi, 'r') as file:
            sequence = file.read().strip().split()
        best_states = hmm.viterbi(sequence)
        print("Most likely hidden states: ", best_states)