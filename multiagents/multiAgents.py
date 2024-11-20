# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get the distance to the nearest food
        minFoodDistance = float('inf')
        for food in newFood.asList():
            distance = util.manhattanDistance(newPos, food)
            if distance < minFoodDistance:
                minFoodDistance = distance

        foodScore = 1.0 / minFoodDistance if minFoodDistance > 0 else 0

        # Get the distance to the nearest ghost
        minGhostDistance = float('inf')
        for ghost in newGhostStates:
            distance = util.manhattanDistance(newPos, ghost.getPosition())
            if distance < minGhostDistance:
                minGhostDistance = distance

        ghostScore = 1.0 / minGhostDistance if minGhostDistance > 0 else 0
        
        ghost_penalty_factor = 1.5 # We introduce a constant to adjust the balance between the importance of food and the penalty for ghosts when combining the scores, the choice of this factor was done through experimentation and observation, and fine-tuning based on the performance of the agent in different scenarios
        return successorGameState.getScore() + foodScore - ghost_penalty_factor * ghostScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentIndex = 0  # Pacman is agent 0

        # Initial call to minimax
        bestAction = self.minimax(gameState, depth, agentIndex)
        return bestAction

    def minimax(self, state, depth, agentIndex):
        # Check if the game is over or if the depth limit is reached
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        # Get legal actions for the current agent
        legalActions = state.getLegalActions(agentIndex)

        if agentIndex == 0:  # Pacman
            maxScore = float('-inf')
            bestAction = None

            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                score = self.minimax(successorState, depth, agentIndex + 1)

                if score > maxScore:
                    maxScore = score
                    bestAction = action

            if depth == 0:  # Return the best action at the root level
                return bestAction
            else:
                return maxScore

        else:  # Ghosts
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            minScore = float('inf')

            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                # If all ghosts have moved, increment the depth
                if agentIndex == state.getNumAgents() - 1:
                    score = self.minimax(successorState, depth + 1, nextAgentIndex)
                else:
                    score = self.minimax(successorState, depth, nextAgentIndex)

                minScore = min(minScore, score)

            return minScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, alpha, beta, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            v = float("-inf")
            bestAction = None

            for action in state.getLegalActions(0):  # Pacman's turn
                successor = state.generateSuccessor(0, action)
                successorValue, _ = minValue(successor, alpha, beta, depth, 1)
                if successorValue > v:
                    v = successorValue
                    bestAction = action
                if v > beta:
                    return v, bestAction
                alpha = max(alpha, v)

            return v, bestAction

        def minValue(state, alpha, beta, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            v = float("inf")
            bestAction = None

            for action in state.getLegalActions(agentIndex):  # Ghost's turn
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:  # Last ghost, reduce depth
                    successorValue, _ = maxValue(successor, alpha, beta, depth - 1)
                else:
                    successorValue, _ = minValue(successor, alpha, beta, depth, agentIndex + 1)

                if successorValue < v:
                    v = successorValue
                    bestAction = action
                if v < alpha:
                    return v, bestAction
                beta = min(beta, v)

            return v, bestAction

        _, bestAction = maxValue(gameState, float("-inf"), float("inf"), self.depth)
        return bestAction
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
