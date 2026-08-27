# LYF Function Plotter V1.0

English | [中文](README.md)

An **LLM-driven (DeepSeek)** interactive function plotting tool for math learning: type a function expression to plot it, or describe transformations like "shift up", "stretch", "reflect" in natural language — or directly ask the model to modify the function. Designed for high-school mathematics study and classroom demonstration.

## ✨ Features

- **Function plotting**: supports `sin(x)`, `x^2+2x-1`, `1/x`, `sqrt(x)`, `log(x)`, etc.; multiple curves at once
- **Smart math notation**: understands `4x`, `x^2`, `x(x+1)`, `2sin(x)`, `2pi` (implicit multiplication)
- **Natural-language AI interaction**: "shift up by 2 units", "stretch vertically by 2x", "reflect about the x-axis" transform the curve; "add 2 to the constant term" makes the LLM generate a new expression
- **Learning tools**: numerical derivative, tangent line (with tangent point and slope), definite integral (shaded area)
- **Interactive view**: wheel/box zoom, pan, reset, save image; curves auto-extend while zooming or panning
- **Safe evaluation**: expressions are parsed with an AST whitelist — only registered math functions are allowed, no raw `eval`

## 📸 Screenshots

![Function transform demo](docs/demo-transform.png)

![Derivative and tangent demo](docs/demo-derivative.png)

## 🚀 Getting Started

### Option 1: Run from source

Requires Python 3.10+. Install dependencies:

```bash
pip install numpy matplotlib requests
```

Run:

```bash
python function_plotter.py
```

On Windows you can also double-click `function_plotter.pyw` (no console window).

### Option 2: Packaged executable

You can build a single-file `.exe` with PyInstaller (or check the Releases page in the future). Double-click to run — no Python installation needed.

## 🤖 Using the AI Features

The first time you use "Transform / Modify / Quick commands", the app asks for your **DeepSeek API Key** (get one at https://platform.deepseek.com).

- The key is saved to `config.json` next to the program
- ⚠️ `config.json` contains your secret key — **never commit it or share it**

Example commands:

| Category | Example |
| --- | --- |
| Translate | shift up by 2 units / shift left by 3 units |
| Rotate | rotate 90° about the origin (counter-clockwise positive) |
| Reflect | about the x-axis / y-axis / origin |
| Stretch | stretch vertically by 2x / compress horizontally by 3x |
| Modify | add 2 to the constant term / change the coefficient to 3 / square the whole expression |
| Tools | derivative / tangent line / definite integral |

## 📁 Project Structure

```
traetest/
├── function_plotter.py   # Main program
├── function_plotter.pyw  # Windows launcher (no console)
├── config.json           # DeepSeek API key (local only, not in the repo)
└── icons/                # Toolbar button icons
```

## 🛡️ Security Notes

- Expressions are evaluated with an AST whitelist: only arithmetic, powers, and registered math functions are allowed
- The API key is stored only in the local `config.json`, which is excluded via `.gitignore`

## 📄 About

A personal learning project by LYF. Feel free to fork it for study and communication.
