# Porymax - Competitive Pokemon AI & Coaching Platform

A state-of-the-art, recursive reinforcement learning agent for Pokémon Showdown's Gen 9 OverUsed (OU) format.

- 142M-parameter transformer policy
- Monte Carlo Tree Search (MCTS) lookahead
- Built-in Flask web platform for wrappers around the model such as Pokemon Showdown bots, Stockfish-style analysis, and team viability scoring.

Porymax also leverages a recursive data flywheel to continuously apply Low-Rank Adaptation (LoRA) and Conservative Q-Learning (CQL) to actively adapt to metagame shifts and new strategies in the Gen 9 OU format.

<details>
<summary><h2 id="setup">Setup</h2></summary>

The entire Porymax project is housed in this single repository. You must have Python 3.10+, Node.js (LTS version recommended), and Git in order to work with Porymax. To set it up locally, first clone the repository:

```shell
git clone https://github.com/ThePeeps191/porymax.git
cd porymax
```

Next, create a virtual environment:

```shell
python -m venv venv

venv\Scripts\activate       # Windows (Powershell)
venv\Scripts\activate.bat   # Windows (Command Prompt)
source venv/bin/activate    # Mac / Linux
```

The remaining commands within **Setup**, as well as the rest of this **README**, will assume that your virtual environment is activated.

The first dependency we will install is PyTorch. **Choose the command that matches your hardware**:

```shell
# Option A: CPU-Only (For local development / PCs without NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Option B: NVIDIA GPU (For cloud training / laddering with CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then, install the rest of Porymax's dependencies from the `requirements.txt` file:

```shell
pip install -r requirements.txt
```

Porymax relies on [Metamon](https://github.com/UT-Austin-RPL/metamon) to run inference and train its AI models, so Metamon must also be cloned and installed:

```shell
mkdir external
cd external
git clone --recursive git@github.com:UT-Austin-RPL/metamon.git
cd metamon
pip install -e .
cd ../..
```

*(Optional but recommended)*: If you have an NVIDIA GPU and want maximum inference speed, you can install FlashAttention 2:

```shell
pip install amago[flash]
```

Kakuna was originally trained using FlashAttention 2, which strictly requires an NVIDIA GPU. If you are running Porymax locally on a CPU (or a GPU without FlashAttention installed), Amago will throw an error. To fix this, **you must run our automated patching script**. It safely modifies the cached config files to use standard PyTorch CPU attention:

```shell
python scripts/patch_attention.py
```

Porymax also uses [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) as a server for hosting and simulating Pokemon battles. Metamon already contains a ready-to-use Pokemon Showdown server:

```shell
cd external/metamon/server/pokemon-showdown
npm install
```

The Pokemon Showdown server must be run in the background while using Porymax locally. A guide on using Porymax with Smogon's official [Pokemon Showdown](https://play.pokemonshowdown.com) server is [below](#online-pokemon-showdown).

```shell
node pokemon-showdown start --no-security
```

</details>

## Quick Start

Once you have Porymax set up (via the [Setup](#setup) instructions above), run the command below to test your installation:

## Online Pokemon Showdown

## License

Porymax is licensed under the MIT License.

## Credits

- **Metamon** for training the base Kakuna model and providing the model inference engine
- **Smogon** for creating the Pokemon Showdown battle simulator
