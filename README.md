# 42 US - Silicon Valley

## Walking Marvin

### Who is Marvin?

Marvin, the Paranoid Android, is a fictional character in
The Hitchhiker's Guide to the Galaxy series by Douglas Adams.
Marvin is the ship's robot aboard the starship Heart of Gold.

### Goals

This is a python project, that uses OpenAI Gym with an environment called Marvin.
The goal is to train Marvin to walk, having the training and walking process.
The total reward for each episode after training is bigger than 100. During the
development, we learned how to use neural networks to help Marvin
get back on his feet, without using any libraries that do the goal of the
project for us, like Evostra or Tensorflow.

### Usage

**Basic form:**

`python marvin.py`

The program display log for each episode.

**Advanced options:**

| Flags               | Description                                                                                   |
| :------------------ |:--------------------------------------------------------------------------------------------- |
| `–-walk (-w)`       | Display only walking process.                                                                 |
| `–-load (-l)`       | Load a set of weights                                                                         |
| `–-save (-s)`       | Save the weights once is trained                                                              |
| `--help (-h)`       | Show flags                                                                                    |
                                       |

*If the program launches without arguments, display training process and walking
process.*

### Setup

Use `sh setup.sh` to setup and build all the dependencies.

*All the dependencies will be installed to the user by running the script...*
