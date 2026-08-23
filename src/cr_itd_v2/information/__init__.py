"""Closed-form linear-Gaussian imaging-information objectives."""

from .local_gaussian import (
    ChannelMomentTargetInformationResult,
    JointLocalTargetInformationResult,
    LocalTargetInformationResult,
    channel_moment_target_information,
    evaluate_channel_moment_target_information,
    evaluate_joint_target_information,
    evaluate_local_target_information,
    joint_target_information,
    local_target_information,
)

__all__ = [
    "ChannelMomentTargetInformationResult",
    "JointLocalTargetInformationResult",
    "LocalTargetInformationResult",
    "channel_moment_target_information",
    "evaluate_channel_moment_target_information",
    "evaluate_joint_target_information",
    "evaluate_local_target_information",
    "joint_target_information",
    "local_target_information",
]
