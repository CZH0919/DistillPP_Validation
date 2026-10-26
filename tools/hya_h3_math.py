"""Flow-matching adaptation of Distillation++ endpoint refinement.

Array-generic pure algebra; model calls and random streams live in the worker.
No approximation alpha(t) ~= alpha(s), no velocity-as-epsilon substitution.
"""
import hashlib


def endpoints(z, velocity, sigma):
    return z - sigma * velocity, z + (1 - sigma) * velocity


def renoise(clean_estimate, independent_noise, query_sigma):
    if not 0 < float(query_sigma) < 1:
        raise ValueError("Teacher query sigma must lie strictly between 0 and 1")
    return (1 - query_sigma) * clean_estimate + query_sigma * independent_noise


def correct_euler(base_next, student_clean, teacher_clean, next_sigma, strength):
    if not 0 <= strength <= 1:
        raise ValueError("Correction strength must be between 0 and 1")
    if strength == 0:
        return base_next
    return base_next + (1 - next_sigma) * strength * (teacher_clean - student_clean)


def renoise_seed(prompt_id, initial_seed, step):
    payload = f"HYA_H3_RENOISE_V1|{initial_seed}|{prompt_id}|{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


