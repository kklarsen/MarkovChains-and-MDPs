import sys
import numpy as np
import random

class Mdp:
    """MARKOV CHAINS AND MARKOV DECISION PROCESS (MDP) TOOLS."""
    
    def __init__(self, states = {}, pi = [], T = [], R = [], EndState = None, gamma = 0.90, epsilon = 0.01):
        
        self.states  = states            # The Markov chain's state descriptions.
        self.pi      = pi                # pi is a given state.
        self.T       = T                 # T is the Markov Chain transition probability matrix.
        self.R       = R                 # Reward vector.
        self.EndState= EndState          # Absorbing state and/or state that ends the simulation.
        
        self.gamma   = gamma             # gamma is the discount factor used in value iterations.
        self.epsilon = epsilon           # epsilon provides the convergence criteria.
        self.ss_check= 0                 # Checks whether mc_steady_state() function has been called.
  
        self.values  = None              # State values.
        self.G       = None              # The long-run average value of a Markov Decision Process (MDP).
        self.policy  = None              # MDP policy.
        self.ctime   = None              # Time to conversion.
       
    def mc_rec_trans(self):
        """ CALCULATE THE RECURRENT & TRANSIENT STATES.
        """
        P = self.T

        j = 1
        while True:
            Pold = P
            P = P**j
            j+=1
            
            P1 = P[P>0]
            P1 = P1[P1<1]
            if P1.size==0: P1 = 0.0
            if P1 <= self.epsilon:
                    break

        recur_states = sum(sum(P==1))
        trans_states = P.shape[0]-recur_states
        
        return recur_states, trans_states

    def mc_stationary_distribution(self):
        """ CHECK THAT ALL ROWS ADD UP TO 1.
        """
        if self.ss_check == 1: 
            p = self.pi
        else:
            p = self.mc_steady_state()
        
        tst1 = np.allclose(p.sum(0), 1, rtol=0.00001, atol=0.00001)
        tst2 = np.allclose(sum(p<0), 0, rtol=0.00001, atol=0.00001)
        tst3 = np.allclose(p.dot(self.T), p, rtol=self.epsilon, atol=self.epsilon)
                     
        tst = np.array([tst1, tst2, tst3])
        tst = (sum(tst)==3)
        
        return tst
    
    def mc_check_prob(self):
        """ CHECK THAT ALL ROWS ADD UP TO 1.
        """
        return self.T.sum(1) == np.ones(self.T.shape[0])

    def mc_absorb(self):
        """ CALCULATE NUMBER OF ABSORBING STATES.
        """
        return sum(np.diagonal(self.T)==1)
        
    def mc_irreducable(self):
        """ CHECK WHETHER MARKOV CHAIN IS IRREDUCABLE.
        """
        return (sum(sum(self.T==0))==0)       
        
    def mc_aperiod(self):
        """ CHECK WHETHER MARKOV CHAIN IS APERIODIC.
        """
        return (sum(np.diagonal(self.T)>0)>0)       
    
    def mc_steady_state(self):
        """ STEADY-STATE ASSESSMENT.
        Finding the Markov Chains steady-state states.
        # pi is a given state.
        # T is the transition probability matrix
        # epsilon provides the convergence criteria
        """

        j = 0 #Counter
        
        p = self.pi
        
        while True:
        
            oldpi = p
    
            p = p.dot(self.T)
                 
            j+=1

            # Check Convergence
            if np.max(np.abs(p - oldpi)) <= self.epsilon:
                break
            # In case of no Convergence
            if j == 1000:
                break
            
            
        self.pi = p
        self.ss_check = 1
        
        return self.pi #Returning the steady-state states.
        
    def mc_characterize(self):
        """ MARKOV CHAIN CHARACTERISTICS.
        """
        absorbing_states = self.mc_absorb()
        aperiodicity_status = self.mc_aperiod()
        stationary_status = self.mc_stationary_distribution()
        reducability_status = self.mc_irreducable()
        probability_status = self.mc_check_prob()
        rec, trans = self.mc_rec_trans()

        #w,v = np.linalg.eig(T)

        print("Total number of states     : ", self.T.shape[0])
        print("All rows of T add up to 1  : ", probability_status)
        print("Number of absorbing states : ", absorbing_states)
        print("Number of recurring states : ", rec)
        print("Number of transient states : ", trans)
        print("Markov Chain Stationary    : ", stationary_status)
        print("Markov Chain Irreducable   : ", reducability_status)
        print("Markov Chain Aperiodic     : ", aperiodicity_status)
        
        return
        
        
    def mc_find_prob(self, seq = []):
        """ PROBABILITY OF A GIVEN STATE SEQUENCE.
        - T is the Markov Chain's Transition Probability Matrix.
        - pi is the steady-state of the Markov chain.
        - seq is the sequence of which the probability is sought.
        """    
        start_state = seq[0]
        prob = self.pi[start_state]
        prev_state = start_state
        
        for i in range(1, len(seq)):
            curr_state = seq[i]
            prob *= self.T[prev_state][curr_state]
            prev_state = curr_state
            
        return prob
        
    def mc_random_walk(self, state0, tsteps):
        """ MONTE CARLO SIMULATION ON MARKOV CHAIN.
        Random walk simulation on the Markov Chain.
        - T is the Markov Chain's Transition Probability Matrix.
        - State0 is the initial state at t = 0.
        - tsteps is the number of time increments considered.
        """    
        n = tsteps
        start_state = state0
        
        print(self.states[start_state],'--->', end = ' ')
        prev_state = start_state

        
        while n-1:
        
            curr_state = np.random.choice(list(self.states.keys()), p = self.T[prev_state])
            print(self.states[curr_state], '--->', end = ' ')
            prev_state = curr_state
            if self.states[prev_state] == self.EndState: break
            n-=1
        print('Ended')
        
        return
        
    def mc_random_walk_reward(self, state0, tsteps):
        """EXPECTED POLICY & VALUE ASSESSMENT VIA RANDOM WALKS
        - states  : Array which includes the names of the n states being studied. 
        - T       : The transition probability matrix of dim (n,n)
        - R       : The reward array of dim (n,) e.g., can have several columns pending #policies.
        - gamma   : Value Discount Factor, < 1.0
        - epsilon : Provides the convergence criteria.
        """
            
        n = tsteps
        stp = 0
        start_state = state0
        path = [self.states[start_state]]
        val_state = self.R[state0]*(self.gamma**0)
        
        prev_state = start_state
        
        i = 1 
        while n:
            curr_state = np.random.choice(list(self.states.keys()), p = self.T[prev_state])
            val_state += self.R[curr_state]*(self.gamma**i)
            
            #print(val_state, R[curr_state])
            
            path+=[self.states[curr_state]]
            prev_state = curr_state
            i+=1
            n-=1
            if self.states[prev_state] == self.EndState: 
                stp = n + 1
                break
        
        self.G = val_state
        self.ctime = stp  
        return (self.G, self.ctime, path)
        
    def mdp_valueIteration(self):
        """ VALUE AND POLICY ITERATION - MARKOV DECISION PROCESS (MDP).
        - states  : Array which includes the names of the n states being studied. 
        - T       : The transition probability matrix of dim (n,n)
        - R       : The reward array of dim (n,) e.g., can have several columns pending #policies.
        - gamma   : Value Discount Factor, < 1.0
        - epsilon : Provides the convergence criteria.
        """
        if (self.states == {}): 
            print('States undefined')
            return(None, None, None, None)
        if (self.R == []):
            print('Rewards undefined')
            return(None, None, None, None)
        
        # Initialize V_0 to zero
        self.values = np.zeros(len(self.states))

        self.ctime = 0
        
        # Value iteration
        # Continue until convergence.
        while True:

            # To be used for convergence check
            oldValues = np.copy(self.values)

            self.values = np.transpose(self.R) + self.gamma*np.dot(self.T,self.values)     # Value iteration step
     
            self.policy = self.values.argmax(0)   # Take the best policy.
            self.values = self.values.max(0)
            
            self.ctime +=1
            
            # Check Convergence
            if np.max(np.abs(self.values - oldValues)) <= self.epsilon:
                break
        
        self.G = self.values.dot(self.pi)
        return(self.values, self.G, self.policy, self.ctime)      
