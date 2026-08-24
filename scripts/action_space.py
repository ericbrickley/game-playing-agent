class ActionSpace:
    """Defines and manages game actions"""
    
    def __init__(self, game_type="hades"):
        self.game_type = game_type
        self.actions = self._get_actions(game_type)
        self.action_names = list(self.actions.keys())
        
    def _get_actions(self, game_type):
        """Get action definitions for game type"""
        if game_type == "hades":
            return {
                "W": {
                    "key": "w",
                    "type": "movement",
                    "description": "Move up",
                    "cooldown": 0,
                    "input_type": "continuous"
                },
                "A": {
                    "key": "a",
                    "type": "movement",
                    "description": "Move left",
                    "cooldown": 0,
                    "input_type": "continuous"
                },
                "S": {
                    "key": "s",
                    "type": "movement",
                    "description": "Move down",
                    "cooldown": 0,
                    "input_type": "continuous"
                },
                "D": {
                    "key": "d",
                    "type": "movement",
                    "description": "Move right",
                    "cooldown": 0,
                    "input_type": "continuous"
                },
                "SPACE": {
                    "key": "space",
                    "type": "attack",
                    "description": "Heavy attack / ability",
                    "cooldown": 0.5,
                    "input_type": "discrete"
                },
                "SHIFT": {
                    "key": "shift",
                    "type": "dash",
                    "description": "Dash / roll",
                    "cooldown": 0.3,
                    "input_type": "discrete"
                },
                "ALT": {
                    "key": "g",
                    "type": "skill",
                    "description": "Ability 1 (remapped from Alt: OS focus trap)",
                    "cooldown": 1.0,
                    "input_type": "discrete"
                },
                "TAB": {
                    "key": "r",
                    "type": "skill",
                    "description": "Ability 2 (remapped from Tab: OS focus trap)",
                    "cooldown": 1.0,
                    "input_type": "discrete"
                },
                "Q": {
                    "key": "q",
                    "type": "skill",
                    "description": "Ability 3",
                    "cooldown": 1.5,
                    "input_type": "discrete"
                },
                "E": {
                    "key": "e",
                    "type": "item",
                    "description": "Use item / interact",
                    "cooldown": 0.2,
                    "input_type": "discrete"
                },
                "F": {
                    "key": "f",
                    "type": "item",
                    "description": "Use item / interact",
                    "cooldown": 0.2,
                    "input_type": "discrete"
                }
            }
        else:
            # Default actions
            return {
                "W": {"key": "w", "type": "movement", "description": "Move up"},
                "A": {"key": "a", "type": "movement", "description": "Move left"},
                "S": {"key": "s", "type": "movement", "description": "Move down"},
                "D": {"key": "d", "type": "movement", "description": "Move right"},
                "SPACE": {"key": "space", "type": "attack", "description": "Attack"},
                "SHIFT": {"key": "shift", "type": "dash", "description": "Dash"},
                "ESC": {"key": "escape", "type": "menu", "description": "Open menu"}
            }
    
    def validate_action(self, action_name):
        """Check whether action name is defined"""
        return action_name in self.actions
        
    def get_action_key(self, action_name):
        """Get keyboard key for action name"""
        return self.actions.get(action_name, {}).get("key", action_name.lower())
        
    def get_action_type(self, action_name):
        """Get action type"""
        return self.actions.get(action_name, {}).get("type", "unknown")
        
    def get_action_description(self, action_name):
        """Get action description"""
        return self.actions.get(action_name, {}).get("description", "")
        
    def get_continuous_actions(self):
        """Get actions that can be held down"""
        return {k: v for k, v in self.actions.items() 
                if v.get("input_type") == "continuous"}
                
    def get_discrete_actions(self):
        """Get actions that are single presses"""
        return {k: v for k, v in self.actions.items() 
                if v.get("input_type") == "discrete"}
        
# Global action space
_action_space = None

def get_action_space(game_type="hades"):
    """Get or create action space instance"""
    global _action_space
    
    if _action_space is None or _action_space.game_type != game_type:
        _action_space = ActionSpace(game_type)
        
    return _action_space
