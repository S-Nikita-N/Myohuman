"""
Wrapper classes for expert policies used in Arnold training.

Experts:
- Kinesis: MyoLegs Walk to point expert
- MyoChallenge: MyoArm Object relocate expert
"""

from myohuman.arnold.experts.expert_wrapper import ExpertWrapper
from myohuman.arnold.experts.kinesis_expert import KinesisExpert
from myohuman.arnold.experts.myochallenge_expert import MyoChallengeExpert

__all__ = [
    "ExpertWrapper",
    "KinesisExpert", 
    "MyoChallengeExpert",
]

