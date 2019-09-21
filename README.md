# RL-ROBO-BOOK-EXAMPLE
同人誌「強化学習基礎から実践 PyTorchとmachinaで二足歩行エージェントをつくる」
のサンプルコードです。

## setup
[poetry](https://github.com/sdispater/poetry)でdependencyの管理をしています。

```bash
$ git clone https://github.com/syundo0730/rl-robo-book-examples.git
$ cd rl-robo-book-examples
$ poetry install
```

### 録画のためのsetup
#### OS X
```bash
$ brew install ffmpeg
```
#### Ubuntu
```bash
$ sudo apt install ffmpeg
```

## 実行
PPO
```bash
$ poetry run python rl_example/run_ppo.py
```

SVG
```bash
$ poetry run python rl_example/run_svg.py
```

AIRL
```bash
$ poetry run python rl_example/run_airl.py
```

Behavior Clone & PPO
```bash
$ poetry run python rl_example/bc_ppo.py
```
