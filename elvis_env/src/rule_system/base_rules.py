#!/usr/bin/env python3
"""
Base Rule System for Video Generation
Provides extensible framework for defining video generation rules
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np


class BaseRule(ABC):
    """Base class for all video generation rules"""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # Higher priority rules apply first
    
    @abstractmethod
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply rule to objects and return modified objects"""
        pass
    
    def should_apply(self, objects: List[Dict], frame_info: Dict) -> bool:
        """Check if rule should be applied (default: always apply)"""
        return True


class ColorRule(BaseRule):
    """Rule for managing object colors"""
    
    def __init__(self, color_mapping: Dict[str, Tuple[int, int, int]] = None, 
                 default_colors: List[Tuple[int, int, int]] = None):
        super().__init__("color_rule", priority=100)
        self.color_mapping = color_mapping or {}
        self.default_colors = default_colors or [
            (100, 150, 255), (255, 150, 100), (150, 255, 100), 
            (255, 100, 150), (150, 100, 255), (100, 255, 150)
        ]
    
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply color rules to objects"""
        for obj in objects:
            if 'color' not in obj or obj['color'] is None:
                # Assign default color based on entity_id or index
                entity_id = obj.get('entity_id', f"obj_{obj.get('id', 0)}")
                if entity_id in self.color_mapping:
                    obj['color'] = self.color_mapping[entity_id]
                else:
                    # Use default colors cyclically
                    color_idx = hash(entity_id) % len(self.default_colors)
                    obj['color'] = self.default_colors[color_idx]
        return objects


class SpeedRule(BaseRule):
    """Rule for managing object-specific speeds"""
    
    def __init__(self, speed_overrides: Dict[str, float] = None):
        super().__init__("speed_rule", priority=60)
        self.speed_overrides = speed_overrides or {}
    
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply speed overrides to specific objects"""
        for obj in objects:
            entity_id = obj.get('entity_id', '')
            if entity_id in self.speed_overrides:
                obj['velocity'] = self.speed_overrides[entity_id]
        return objects


class MovementRule(BaseRule):
    """Rule for managing object movement patterns"""
    
    def __init__(self, movement_patterns: Dict[str, Dict] = None):
        super().__init__("movement_rule", priority=50)
        self.movement_patterns = movement_patterns or {}
    
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply movement rules to objects"""
        for obj in objects:
            pattern_key = self._get_pattern_key(obj)
            if pattern_key in self.movement_patterns:
                pattern = self.movement_patterns[pattern_key]
                self._apply_movement_pattern(obj, pattern, frame_info)
        return objects
    
    def _get_pattern_key(self, obj: Dict) -> str:
        """Get movement pattern key for an object"""
        # Default: use side or entity_id
        return obj.get('side', obj.get('entity_id', 'default'))
    
    def _apply_movement_pattern(self, obj: Dict, pattern: Dict, frame_info: Dict):
        """Apply specific movement pattern to object"""
        velocity = pattern.get('velocity', obj.get('velocity', 2))
        direction = pattern.get('direction', 'down')
        
        if direction == 'down':
            obj['y'] += velocity
        elif direction == 'up':
            obj['y'] -= velocity
        elif direction == 'left':
            obj['x'] -= velocity
        elif direction == 'right':
            obj['x'] += velocity
        elif direction == 'down_right':
            obj['y'] += velocity * 0.707  # cos(45°)
            obj['x'] += velocity * 0.707  # sin(45°)
        elif direction == 'down_left':
            obj['y'] += velocity * 0.707
            obj['x'] -= velocity * 0.707
        elif direction == 'up_right':
            obj['y'] -= velocity * 0.707
            obj['x'] += velocity * 0.707
        elif direction == 'up_left':
            obj['y'] -= velocity * 0.707
            obj['x'] -= velocity * 0.707
        elif direction == 'down_right':
            obj['y'] += velocity * 0.707  # cos(45°)
            obj['x'] += velocity * 0.707  # sin(45°)
        elif direction == 'down_left':
            obj['y'] += velocity * 0.707
            obj['x'] -= velocity * 0.707
        elif direction == 'up_right':
            obj['y'] -= velocity * 0.707
            obj['x'] += velocity * 0.707
        elif direction == 'up_left':
            obj['y'] -= velocity * 0.707
            obj['x'] -= velocity * 0.707
        elif direction == 'custom':
            # Allow custom movement function
            custom_func = pattern.get('custom_function')
            if custom_func:
                custom_func(obj, frame_info)


class SpeedRule(BaseRule):
    """Rule for managing object-specific speeds"""
    
    def __init__(self, speed_overrides: Dict[str, float] = None):
        super().__init__("speed_rule", priority=60)
        self.speed_overrides = speed_overrides or {}
    
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply speed overrides to specific objects"""
        for obj in objects:
            entity_id = obj.get('entity_id', '')
            if entity_id in self.speed_overrides:
                obj['velocity'] = self.speed_overrides[entity_id]
        return objects


class PlacementRule(BaseRule):
    """Rule for managing object cross-placement scenarios"""
    
    def __init__(self, placement_config: Dict = None):
        super().__init__("placement_rule", priority=80)
        self.placement_config = placement_config or {}
    
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply cross-placement rules to objects"""
        if not self.placement_config.get('enabled', False):
            return objects
        
        cross_placements = self.placement_config.get('cross_placements', [])
        
        for placement in cross_placements:
            source_type = placement.get('source_type')  # e.g., 'left_1', 'right_2'
            target_side = placement.get('target_side')   # e.g., 'right', 'left'
            target_movement = placement.get('target_movement')  # e.g., 'up', 'down'
            
            # Find objects matching source_type
            for obj in objects:
                if obj.get('entity_id') == source_type:
                    # Update object's side and movement behavior
                    obj['side'] = target_side
                    obj['cross_placed'] = True
                    obj['cross_movement'] = target_movement
        
        return objects


class InterventionRule(BaseRule):
    """Rule for applying interventions to specific objects"""
    
    def __init__(self, intervention_config: Dict = None):
        super().__init__("intervention_rule", priority=10)
        self.intervention_config = intervention_config or {}
    
    def apply(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply interventions based on configuration"""
        if not self.intervention_config.get('enabled', False):
            return objects
        
        intervention_type = self.intervention_config.get('type', 'reverse_movement')
        target_indices = self.intervention_config.get('target_indices', {})
        
        for obj in objects:
            if self._should_intervene(obj, target_indices):
                self._apply_intervention(obj, intervention_type, frame_info)
        
        return objects
    
    def _should_intervene(self, obj: Dict, target_indices: Dict) -> bool:
        """Check if object should have intervention applied"""
        side = obj.get('side', 'default')
        creation_order = obj.get('creation_order', 0)
        
        if side in target_indices:
            return creation_order in target_indices[side]
        return False
    
    def _apply_intervention(self, obj: Dict, intervention_type: str, frame_info: Dict):
        """Apply specific intervention to object"""
        if intervention_type == 'reverse_movement':
            # Reverse the object's movement direction
            obj['intervention_reverse'] = True
        elif intervention_type == 'freeze':
            obj['velocity'] = 0
        elif intervention_type == 'change_color':
            obj['color'] = self.intervention_config.get('new_color', (255, 0, 0))
        elif intervention_type == 'cross_place':
            # Handle cross-placement intervention
            if obj.get('side') == 'left':
                obj['side'] = 'right'
                obj['cross_placed'] = True
            elif obj.get('side') == 'right':
                obj['side'] = 'left'
                obj['cross_placed'] = True


class RuleEngine:
    """Engine that manages and applies all rules"""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule: BaseRule):
        """Add a rule to the engine"""
        self.rules.append(rule)
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def apply_all_rules(self, objects: List[Dict], frame_info: Dict) -> List[Dict]:
        """Apply all rules in priority order"""
        for rule in self.rules:
            if rule.should_apply(objects, frame_info):
                objects = rule.apply(objects, frame_info)
        return objects
    
    def get_rules(self) -> List[BaseRule]:
        """Get all rules sorted by priority"""
        return self.rules.copy()