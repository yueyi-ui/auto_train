"""Config validator: clamps proposals into the search space and rejects duplicates."""


class Validator:
    def __init__(self, search_space, defaults):
        self.space = search_space
        self.defaults = defaults

    def defaults_config(self):
        return {
            "lr0": self.defaults["lr0"],
            "batch": self.defaults["batch"],
            "epochs": self.defaults["epochs"],
            "imgsz": self.defaults["imgsz"],
            "warmup": self.defaults["warmup"],
            "patience": self.defaults["patience"],
            "workers": self.defaults["workers"],
            "amp": self.defaults["amp"],
        }

    def validate(self, raw, history):
        if not isinstance(raw, dict):
            return self.defaults_config(), False
        out = {}
        for key, spec in self.space.items():
            val = raw.get(key, self.defaults.get(key))
            if isinstance(spec, dict):
                val = self._clamp_number(val, spec["min"], spec["max"])
                if key in ("batch", "epochs", "imgsz", "warmup", "patience", "workers"):
                    val = int(round(val))
            elif isinstance(spec, list):
                val = self._nearest_choice(val, spec)
            out[key] = val
        for t in history:
            if t.get("config") == out:
                return out, False
        return out, True

    @staticmethod
    def _clamp_number(val, lo, hi):
        try:
            return min(max(float(val), lo), hi)
        except (TypeError, ValueError):
            return lo

    @staticmethod
    def _nearest_choice(val, choices):
        try:
            val = float(val)
        except (TypeError, ValueError):
            return choices[0]
        return min(choices, key=lambda c: abs(float(c) - val))
