import os
import sys
import csv
import glob
import argparse
from datetime import datetime
import numpy as np
from threading import Thread, active_count, Event
from time import sleep


sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

from common import parse_clients_args, ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder
from malmopy.environment.malmo import MalmoALEStateBuilder
from malmopy.agent import TemporalMemory, RandomAgent
from malmopy.visualization import ConsoleVisualizer


try:
    from MyDDQNAgent import MyDDQN_Q, PyTorchQModel
    USING_PYTORCH = True
except ImportError:
    USING_PYTORCH = False
    print("worning: PyTorch cannot be imported, using CNTK instead")
try:
    from malmopy.model.cntk import QNeuralNetwork
    from MyDQNAgent import MyDDQN_Q as CntkDDQN_Q
    USING_CNTK = True
except ImportError:
    USING_CNTK = False
    if not USING_PYTORCH:
        print("worning: CNTK cannot be imported, please check your environment")


DEFAULT_CLIENTS = ['127.0.0.1:10000', '127.0.0.1:10001']
NUM_EPISODES = 50

class ModelTester:
    def __init__(self, models_folder, backend='pytorch', clients=DEFAULT_CLIENTS, 
                 device=-1, results_file=None):
        """
        initialize the ModelTester with the given parameters.
        
        Args:
            models_folder: model files folder path
            backend: 'pytorch' or 'cntk'
            clients: Minecraft client endpoints (ip:port)
            device: GPU device index (-1 for CPU)
            results_file: output CSV file path
        """
        self.models_folder = models_folder
        self.backend = backend
        self.clients = parse_clients_args(clients)
        self.device = device
        
        if results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_file = f"model_test_results_{timestamp}.csv"
        else:
            self.results_file = results_file
            
        
        self.results = {}
        self.stop_event = Event()
        
        
        self.visualizer = ConsoleVisualizer()

    def find_models(self):

        """find all model files in the specified folder"""
        if self.backend == 'pytorch':
            models = glob.glob(os.path.join(self.models_folder, "*.pt"))
        else:  # cntk
            models = glob.glob(os.path.join(self.models_folder, "*.model"))
        
    
        def extract_steps(model_path):
            """from the model file name, extract the training steps"""
            filename = os.path.basename(model_path)
            try:
                # get steps from the file name
                # match patterns like "pig_chase-dqn_fast_465000.model" or "pig_chase-dqn_fast_465000.pt"

                step_str = filename.split('_')[-1].split('.')[0]
                return int(step_str)
            except (IndexError, ValueError):

                print(f"worning: could not extract steps from {filename}, using 0")
                return 0
        
        # loop through the models and print their names
        sorted_models = sorted(models, key=extract_steps)
        print(f"find {len(sorted_models)} models in {self.models_folder}:")

        sorted_models = sorted(models, key=extract_steps)
        
        sample_interval = 2  # sample every 2 models


        if sample_interval > 1 and len(sorted_models) > sample_interval:
            # always include the first and last model
            first_model = sorted_models[0]
            last_model = sorted_models[-1]
            
            sampled_models = [first_model] + sorted_models[1:-1:sample_interval] + [last_model]
            
            if len(sampled_models) == 2 and first_model != last_model:
                middle_index = len(sorted_models) // 2
                sampled_models.insert(1, sorted_models[middle_index])
        else:
            sampled_models = sorted_models
        
        # Print information about found models
        print(f"Using interval {sample_interval}, selected {len(sampled_models)} models for evaluation:")
        print("Models to be used:")
        for model in sampled_models:
            steps = extract_steps(model)
            print(f"  - {os.path.basename(model)} (steps: {steps})")
        
        return sampled_models

    
    def run_challenge_agent(self, quit_event):
        """
        Run challenge agent (random/focused)
        
        Args:
            quit_event: Quit event for thread synchronization
        """
        name = ENV_AGENT_NAMES[0]
        print(f"Starting challenge agent: {name}")
        
        # Initialize environment with symbolic state builder
        builder = PigChaseSymbolicStateBuilder()
        env = PigChaseEnvironment(
            self.clients,
            builder,
            role=0,  # Challenge agent role
            human_speed=False,
            randomize_positions=True
        )
        
        # Create challenge agent
        agent = PigChaseChallengeAgent(name)
        
        # Determine agent type
        if isinstance(agent.current_agent, RandomAgent):
            agent_type = PigChaseEnvironment.AGENT_TYPE_1
        else:
            agent_type = PigChaseEnvironment.AGENT_TYPE_2
        
        # Initialize reset
        obs = env.reset(agent_type)
        reward = 0
        agent_done = False
        
        # Main agent loop
        while not quit_event.is_set():
            # If episode ends
            if env.done:
                # Re-determine agent type
                if isinstance(agent.current_agent, RandomAgent):
                    agent_type = PigChaseEnvironment.AGENT_TYPE_1
                else:
                    agent_type = PigChaseEnvironment.AGENT_TYPE_2
                    
                # Reset environment
                obs = env.reset(agent_type)
                while obs is None and not quit_event.is_set():
                    print('Warning: Received obs=None, resetting again.')
                    obs = env.reset(agent_type)
            
            # Select action and execute
            action = agent.act(obs, reward, agent_done, is_training=False)
            obs, reward, agent_done = env.do(action)
        
        print(f"Challenge agent {name} has stopped")
    
    def test_model(self, model_path, quit_event):
        """
        Test a model and return average reward
        
        Args:
            model_path: Path to model file
            quit_event: Quit event for thread synchronization
            
        Returns:
            List of rewards for each episode
        """
        name = ENV_AGENT_NAMES[1]
        print(f"Starting model test: {os.path.basename(model_path)}")
        
        # Initialize environment
        env = PigChaseEnvironment(
            self.clients,
            MalmoALEStateBuilder(),
            role=1,  # DQN agent role
            human_speed=False,
            randomize_positions=True
        )
        
        # Create replay memory
        memory = TemporalMemory(100000, (84, 84))
        
        # Create model
        if self.backend == 'pytorch' and USING_PYTORCH:
            model = PyTorchQModel((memory.history_length, 84, 84), env.available_actions)
        elif self.backend == 'cntk' and USING_CNTK:
            model = QNeuralNetwork((memory.history_length, 84, 84), env.available_actions, self.device)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Load model
        try:
            model.load(model_path)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {model_path}, error: {e}")
            return []
        
        # Create agent (evaluation mode)
        if self.backend == 'pytorch' and USING_PYTORCH:
            agent = MyDDQN_Q(
                name=name,
                actions=env.available_actions,
                model=model,
                memory=memory,
                gamma=0.99,
                batch_size=32,
                target_network_train_frequency=1000,
                explorer=None,  # No explorer needed for evaluation mode
                visualizer=self.visualizer,
                backend=self.backend,
                is_evaluating=True,
                My_train_after=10000,
                train_frequency=8,
                tau=1.0
            )
        else:  # cntk
            agent = CntkDDQN_Q(
                name=name,
                actions=env.available_actions,
                model=model,
                memory=memory,
                gamma=0.99,
                batch_size=32,
                target_network_train_frequency=1000,
                explorer=None,  # No explorer needed for evaluation mode
                visualizer=self.visualizer,
                backend=self.backend,
                My_train_after=10000,
                is_evaluating=True,
                train_frequency=8,
                tau=1.0
            )
        
        # Run test
        episode_rewards = []
        current_episode_rewards = []
        episodes_completed = 0
        
        # Initialize
        obs = env.reset()
        reward = 0
        agent_done = False
        
        print(f"Starting test of {NUM_EPISODES} episodes")
        
        # Test loop
        while episodes_completed < NUM_EPISODES and not quit_event.is_set():
            # If episode ends
            if env.done:
                # Record total reward for this episode
                if current_episode_rewards:
                    episode_reward = sum(current_episode_rewards)
                    episode_rewards.append(episode_reward)
                    print(f"Episode {episodes_completed + 1} completed, reward: {episode_reward}")
                    episodes_completed += 1
                
                # Reset
                current_episode_rewards = []
                obs = env.reset()
                while obs is None and not quit_event.is_set():
                    print('Warning: Received obs=None, resetting again.')
                    obs = env.reset()
            
            # Select action and execute
            action = agent.act(obs, reward, agent_done, is_training=False)
            obs, reward, agent_done = env.do(action)
            current_episode_rewards.append(reward)
        
        print(f"Model {os.path.basename(model_path)} testing completed, finished {episodes_completed} episodes")
        return episode_rewards
    
    def run_tests(self):
        """Run tests for all models and collect results"""
        # Find all models
        model_files = self.find_models()
        if not model_files:
            print("No model files found, exiting test")
            return
        
        # Test each model
        for model_file in model_files:
            model_name = os.path.basename(model_file)
            print(f"\nStarting test for model: {model_name}")
            
            # Set up quit event
            quit_event = Event()
            
            # Start challenge agent thread
            challenge_thread = Thread(
                target=self.run_challenge_agent,
                args=(quit_event,)
            )
            challenge_thread.daemon = True
            challenge_thread.start()
            
            # Give server time to start
            sleep(2)
            
            # Run test
            try:
                rewards = self.test_model(model_file, quit_event)
                self.results[model_name] = rewards
                
                # Print results
                if rewards:
                    avg_reward = np.mean(rewards)
                    print(f"\nModel {model_name} test results:")
                    print(f"  Episodes completed: {len(rewards)}")
                    print(f"  Average reward: {avg_reward:.2f}")
                    print(f"  Maximum reward: {max(rewards) if rewards else 'N/A'}")
                    print(f"  Minimum reward: {min(rewards) if rewards else 'N/A'}")
                else:
                    print(f"\nModel {model_name} did not complete any episodes")
            
            finally:
                # Stop challenge agent
                quit_event.set()
                challenge_thread.join(timeout=5)
                
                # Wait some time to ensure environment fully closes
                sleep(2)
                
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save test results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        # Determine CSV columns - find model with most completed episodes
        max_episodes = max(len(rewards) for rewards in self.results.values())
        
        # Prepare CSV rows
        rows = []
        for model_name, rewards in self.results.items():
            # Pad insufficient episodes to align all models
            padded_rewards = rewards + [''] * (max_episodes - len(rewards))
            rows.append([model_name] + padded_rewards)
        
        # Create CSV header row
        header = ['Model'] + [f'Reward_{i+1}' for i in range(max_episodes)]
        
        # Write to CSV file
        with open(self.results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"Results saved to: {self.results_file}")
        
        # Additionally write a CSV with summary statistics
        summary_file = os.path.splitext(self.results_file)[0] + "_summary.csv"
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Episodes_Completed', 'Avg_Reward', 'Max_Reward', 'Min_Reward'])
            
            for model_name, rewards in self.results.items():
                if rewards:
                    writer.writerow([
                        model_name,
                        len(rewards),
                        np.mean(rewards),
                        max(rewards),
                        min(rewards)
                    ])
                else:
                    writer.writerow([model_name, 0, 'N/A', 'N/A', 'N/A'])
        
        print(f"Summary statistics saved to: {summary_file}")

def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(description='Pig Chase DDQN Model Tester')
    
    parser.add_argument(
        '-f', '--folder',
        type=str,
        required=True,
        help='Path to folder containing model files'
    )
    
    parser.add_argument(
        '-b', '--backend',
        type=str,
        choices=['cntk', 'pytorch'],
        default='pytorch',
        help='Neural network backend'
    )
    
    parser.add_argument(
        '-c', '--clients',
        nargs='*',
        default=DEFAULT_CLIENTS,
        help='Minecraft client endpoints (ip:port)'
    )
    
    parser.add_argument(
        '-d', '--device',
        type=int,
        default=-1,
        help='GPU device index (-1 for CPU)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '-e', '--episodes',
        type=int,
        default=50,
        help='Number of episodes to test each model'
    )
    
    args = parser.parse_args()
    
    # Update global constant
    global NUM_EPISODES
    NUM_EPISODES = args.episodes
    
    # Validate folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return 1
    
    # Create and run tester
    tester = ModelTester(
        models_folder=args.folder,
        backend=args.backend,
        clients=args.clients,
        device=args.device,
        results_file=args.output
    )
    
    print(f"Starting tests for models in folder '{args.folder}'")
    print(f"Each model will be tested for {NUM_EPISODES} episodes")

    # Set up backend
    if args.backend == 'cntk':
        from malmopy.model.cntk import QNeuralNetwork
        from MyDQNAgent import MyDDQN_Q as CntkDDQN_Q

    elif args.backend == 'pytorch' :
        from MyDDQNAgent import MyDDQN_Q, PyTorchQModel

    tester.run_tests()
    return 0

if __name__ == "__main__":
    sys.exit(main())


 # py -3.6 Myevaluation.py -f testmodel -b cntk
 # py -3.6 Myevaluation.py -f testmodel -b pytorch