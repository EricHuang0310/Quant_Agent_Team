import os
from pathlib import Path
from typing import Dict, Optional

from utils.logger import setup_logger

logger = setup_logger("utils.prompt_loader")

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptLoader:
    """Loads .md prompt files with {{variable}} template substitution and hot-reload.

    Hot-reload: on each ``load()`` call the file's mtime is checked.
    If the file was modified since last read, it is re-read from disk automatically.
    This lets you tweak prompts during a live trading session without restarting.
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        hot_reload: bool = True,
    ):
        self._dir = Path(prompts_dir) if prompts_dir else _DEFAULT_PROMPTS_DIR
        self._cache: Dict[str, str] = {}
        self._mtimes: Dict[str, float] = {}
        self._hot_reload = hot_reload

    def load(
        self,
        filename: str,
        variables: Optional[Dict[str, str]] = None,
    ) -> str:
        """Load a .md prompt file and substitute ``{{key}}`` placeholders.

        Parameters
        ----------
        filename : str
            Name of the file inside the prompts directory (e.g. ``"meta_agent.md"``).
        variables : dict, optional
            Mapping of placeholder names to replacement values.

        Returns
        -------
        str
            The prompt content with all placeholders replaced.
        """
        filepath = self._dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        mtime = os.path.getmtime(filepath)

        needs_reload = (
            filename not in self._cache
            or (self._hot_reload and mtime > self._mtimes.get(filename, 0))
        )

        if needs_reload:
            with open(filepath, "r", encoding="utf-8") as f:
                self._cache[filename] = f.read()
            self._mtimes[filename] = mtime
            logger.debug(f"Loaded prompt: {filename}")

        content = self._cache[filename]

        if variables:
            for key, value in variables.items():
                content = content.replace("{{" + key + "}}", str(value))

        return content

    def load_all_strategy_descriptions(self) -> Dict[str, str]:
        """Load all strategy agent .md files and return ``{agent_name: content}``.

        These descriptions are fed into the Meta-Agent prompt so it understands
        each strategy's strengths, weaknesses, and parameters.
        """
        descriptions: Dict[str, str] = {}
        strategy_files = [
            "momentum_agent.md",
            "mean_reversion_agent.md",
            "grid_agent.md",
            "execution_agent.md",
        ]
        for fname in strategy_files:
            path = self._dir / fname
            if path.exists():
                name = fname.replace("_agent.md", "")
                descriptions[name] = self.load(fname)
        return descriptions
