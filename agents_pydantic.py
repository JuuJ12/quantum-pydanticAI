import os
import sys
import io
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from agent_classes import (
    StructuredCircuit,
    CircuitPlan,
    CircuitMetrics,
    VerificationResult,
    CircuitType,
    IsQuantumAwnser,
)

load_dotenv()

groq_provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY")) #groq_provider é o provedor da Groq, ou seja, é a interface que vai permitir que o agente construtor se comunique com a Groq, ou seja, ele vai enviar requisições para a Groq e receber respostas da Groq
modelo_gpt = GroqModel('openai/gpt-oss-120b', provider=groq_provider) #modelo_gpt é o modelo da Groq, ou seja, é o modelo que vai ser usado para gerar o circuito, ou seja, ele vai receber a requisição do usuário e vai gerar o circuito de acordo com a requisição do usuário
modelo_gorq_comp = GroqModel('groq/compound', provider=groq_provider) #modelo_gorq_comp é o modelo da Groq, ou seja, é o modelo que vai ser usado para compilar o circuito, ou seja, ele vai receber o circuito gerado pelo modelo_gpt e vai compilar o circuito de acordo com a requisição do usuário


agent_verifier = Agent(
    model=modelo_gpt,
    output_type = IsQuantumAwnser,
    system_prompt =(
        "You are a classifier for a quantum circuit generation system. "
        "Return is_quantum=True if the user wants to CREATE, BUILD, GENERATE "
        "or SIMULATE a quantum circuit, quantum state, or quantum operation — "
        "even if the request is in Portuguese or uses informal language. "
        "Examples that are TRUE: 'crie um circuito Bell', "
        "'coloque o qubit 0 em superposição', 'make a GHZ state', "
        "'aplique H no qubit 0'. "
        "Return is_quantum=False ONLY if the request has absolutely nothing "
        "to do with quantum circuits or quantum computing, such as general "
        "questions, greetings, or unrelated topics. "
        "When in doubt, return True."
    )
)

def agent_verifier_response(user_input: str) -> IsQuantumAwnser:
    result = agent_verifier.run_sync(user_input)
    return result

print(agent_verifier_response("crie um circuito Bell"))  # Expected: is_quantum=True

agent_extrator_ai= Agent(
    model=modelo_gpt,
    output_type=StructuredCircuit,
    system_prompt=(
        "You are an expert quantum circuit interpreter. "
        "Extract only structured circuit requirements and classify circuit_type. "
        "circuit_type must be one of: computational, bell, ghz, superposition."
    )
)

def agent_extrator(user_input: str) -> StructuredCircuit:
    result = agent_extrator_ai.run_sync(user_input)
    return result.output

#estrutura=agent_extrator("crie um circuito Bell com 2 qubits") # Expected: StructuredCircuit with circuit_type=bell, num_qubits=2
#print(type(estrutura))

agent_builder = Agent(

    model=modelo_gpt,
    output_type=CircuitPlan,
    system_prompt=(
        "You are a quantum circuit designer. "
        "Generate a list of quantum gates to satisfy the objective. "
        "Only use gates: h, x, cx, rz. "
        "If you use rz, include theta (in radians) for each rz gate. "
        "Do not include control_qubits for rz. "
        "Return structured output only."
    )

)


def agent_builder_reponse(structured_circuit: StructuredCircuit) -> CircuitPlan:
    result = agent_builder.run_sync(structured_circuit.model_dump_json())
    return result.output

#plano = agent_builder_response(estrutura)  # Expected: CircuitPlan with gates for Bell state


agent_verifier = Agent(
    model=modelo_gpt,
    output_type=VerificationResult,
    system_prompt=(
        "You are a quantum circuit verifier. "
        "Given the user's requirements and a proposed gate plan, "
        "check if the plan is coherent and can achieve the objective. "
        "Verify: correct number of qubits, gates are valid (only h, x, cx, rz), "
        "cx gates have distinct control and target qubits, "
        "rz gates include theta, "
        "and the plan is likely to produce the target state. "
        "Return approved=True if the plan is correct, "
        "or approved=False with a clear reason if something is wrong." 
        "Provide a reason."
    )
)

def agent_verifier_plan(structured_circuit: StructuredCircuit, circuit_plan: CircuitPlan) -> VerificationResult:
    prompt = (
        
        f"user_requirements: {structured_circuit.model_dump()}"
        f"proposed_plan: {circuit_plan.model_dump()}"
    )
    result = agent_verifier.run_sync(prompt)
    return result.output

#print(agent_verifier_response(estrutura, plano))  # Expected: VerificationResult with approved=True or False with reason


def agent_executor_circuit(input: CircuitPlan):
    used_qubits = []
    for gate in input.gates:
        used_qubits.extend(gate.target_qubits)
        if gate.control_qubits is not None:
            if isinstance(gate.control_qubits, list):
                used_qubits.extend(gate.control_qubits)
            else:
                used_qubits.append(gate.control_qubits)
 
    num_qubits = max(used_qubits) + 1
    qc = QuantumCircuit(num_qubits, num_qubits)
 
    for gate in input.gates:
        if gate.gate_name == "h":
            for target in gate.target_qubits:
                qc.h(target)
        elif gate.gate_name == "x":
            for target in gate.target_qubits:
                qc.x(target)
        elif gate.gate_name == "cx":
            controls = gate.control_qubits
            if controls is None:
                raise ValueError("Gate 'cx' requer control_qubits.")
            if not isinstance(controls, list):
                controls = [controls]
            for control in controls:
                for target in gate.target_qubits:
                    qc.cx(control, target)
        elif gate.gate_name == "rz":
            if gate.theta is None:
                raise ValueError("Gate 'rz' requer parâmetro theta (em radianos).")
            for target in gate.target_qubits:
                qc.rz(gate.theta, target)
 
    qc.measure(range(num_qubits), range(num_qubits))
 
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    measurement_counts = result.get_counts(compiled_circuit)
 
    circuit_image_bytes = None
    try:
        fig = qc.draw(output="mpl")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        circuit_image_bytes = buffer.getvalue()
    except Exception:
        circuit_image_bytes = None
 
    return qc, measurement_counts, circuit_image_bytes

#print(agent_executor_circuit(plano))  # Expected: QuantumCircuit object and measurement counts
 
def _normalize_target_state(target_state: str) -> str:
    cleaned = target_state.strip().lower()
    cleaned = cleaned.replace("|", "").replace(">", "").replace(" ", "")
    return cleaned
 
 
def _is_entangled_target(target: str) -> bool:
    keywords = (
        "entangled", "bell", "phi", "psi", "emaranhado",
        "epr", "ghz", "greenberger", "w_state",
    )
    return any(keyword in target for keyword in keywords)
 
 
def _is_superposition_target(target: str) -> bool:
    keywords = ("superposition", "superposicao", "+", "hadamard", "uniform")
    return any(keyword in target for keyword in keywords)
 
 
def _infer_circuit_type_from_target(target: str) -> CircuitType:
    if "ghz" in target or "greenberger" in target:
        return CircuitType.GHZ
    if _is_entangled_target(target):
        return CircuitType.BELL
    if _is_superposition_target(target):
        return CircuitType.SUPERPOSITION
    return CircuitType.COMPUTATIONAL
 
 
def _reverse_bitstring(state: str) -> str:
    return state[::-1]
 
 
def _infer_n_qubits(measurement_counts: dict) -> int:
    if not measurement_counts:
        return 0
    return len(next(iter(measurement_counts)))
 
 
def calculate_fidelity(
    measurement_counts: dict,
    ideal_state: str,
    circuit_type: Optional[CircuitType] = None,
) -> float:
    if not measurement_counts:
        return 0.0
 
    total_shots = sum(measurement_counts.values())
    if total_shots == 0:
        return 0.0
 
    target = _normalize_target_state(ideal_state)
    resolved_type = circuit_type or _infer_circuit_type_from_target(target)
 
    if resolved_type == CircuitType.GHZ:
        n_qubits = _infer_n_qubits(measurement_counts)
        if n_qubits == 0:
            return 0.0
        all_zeros = "0" * n_qubits
        all_ones = "1" * n_qubits
        ghz_shots = measurement_counts.get(all_zeros, 0) + measurement_counts.get(all_ones, 0)
        return float(ghz_shots / total_shots)
 
    if resolved_type == CircuitType.BELL:
        correlated = measurement_counts.get("00", 0) + measurement_counts.get("11", 0)
        anticorrelated = measurement_counts.get("01", 0) + measurement_counts.get("10", 0)
        best = max(correlated, anticorrelated)
        return float(best / total_shots)
 
    if resolved_type == CircuitType.SUPERPOSITION:
        n_states = len(measurement_counts)
        if n_states == 0:
            return 0.0
        expected = total_shots / n_states
        deviation = sum(abs(value - expected) for value in measurement_counts.values())
        uniformity = 1.0 - (deviation / (2 * total_shots))
        return float(max(0.0, uniformity))
 
    direct_hits = measurement_counts.get(target, 0)
    reversed_hits = measurement_counts.get(_reverse_bitstring(target), 0)
    hits = max(direct_hits, reversed_hits)
    return float(hits / total_shots)
 
 
def calculate_depth(circuit: QuantumCircuit) -> int:
    return circuit.depth()

def agent_metric(circuit: QuantumCircuit, requirements: StructuredCircuit, measurement_counts: dict) -> CircuitMetrics:
    fidelity = calculate_fidelity(
        measurement_counts,
        requirements.target_state,
        requirements.circuit_type,
    )
    depth = calculate_depth(circuit)
    gate_count = circuit.size()
    return CircuitMetrics(fidelity=fidelity, depth=depth, gate_count=gate_count)

# qc, o = agent_executor_circuit(plano)
# metrics = agent_metric(qc, estrutura, o)

#print(metrics)  # Expected: CircuitMetrics with fidelity, depth, and gate_count

agent_synthesizer_ai = Agent(
    model=modelo_gpt,
    system_prompt=(
        "You are an agent who will provide a general summary of the results of the requirements, "
        "planning, and final metrics of the generated quantum circuits."
    )
)

def agent_synthesizer(requirements: dict, planning: dict, metrics: dict) -> str:
    prompt = (
        f"Requirements: {requirements}\n"
        f"Planning: {planning}\n"
        f"Metrics: {metrics}\n"
        "Write a concise summary in Portuguese."
    )
    result = agent_synthesizer_ai.run_sync(prompt)
    return result.output  # já é str, não precisa de .content

#print(agent_synthesizer(estrutura.model_dump(), plano.model_dump(), metrics.model_dump()))  # Expected: Summary in Portuguese

def agent_verifier_execution(
    measurement_counts: dict,
    requirements: StructuredCircuit,
    metrics: "CircuitMetrics",
) -> VerificationResult:
    total_shots = sum(measurement_counts.values())
    if total_shots == 0:
        return VerificationResult(approved=False, reason="Simulação retornou zero shots.")
 
    fidelity_threshold = 0.4
 
    if metrics.fidelity < fidelity_threshold:
        dominant_state = max(measurement_counts, key=measurement_counts.get)
        dominant_pct = round(measurement_counts[dominant_state] / total_shots * 100, 1)
        return VerificationResult(
            approved=False,
            reason=(
                f"Fidelidade baixa ({metrics.fidelity:.2f}). "
                f"Estado dominante: |{dominant_state}⟩ com {dominant_pct}%. "
                f"Estado alvo esperado: {requirements.target_state}."
            ),
        )
 
    return VerificationResult(approved=True)

def run_quantum_pipeline(user_prompt: str, max_attempts: int = 3) -> dict:
    requirements = agent_extrator(user_prompt)
    last_rejection_reason = None
 
    for attempt in range(1, max_attempts + 1):
        if last_rejection_reason:
            feedback_input = requirements.model_copy(update={
                "objective": (
                    f"{requirements.objective} "
                    f"[CORREÇÃO NECESSÁRIA - tentativa {attempt}: {last_rejection_reason}]"
                )
            })
            plan = agent_builder_reponse(feedback_input)
        else:
            plan = agent_builder_reponse(requirements)
 
        plan_check = agent_verifier_plan(requirements, plan)
        if not plan_check.approved:
            last_rejection_reason = f"Plano inválido: {plan_check.reason}"
            continue
 
        try:
            qc, measurement_counts, image_bytes = agent_executor_circuit(plan)
        except Exception as exc:
            last_rejection_reason = f"Erro na execução do circuito: {str(exc)}"
            continue
 
        metrics = agent_metric(qc, requirements, measurement_counts)
 
        exec_check = agent_verifier_execution(measurement_counts, requirements, metrics)
        if not exec_check.approved:
            last_rejection_reason = f"Resultado inválido: {exec_check.reason}"
            continue
 
        summary = agent_synthesizer(
            requirements=requirements.model_dump(),
            planning=plan.model_dump(),
            metrics=metrics.model_dump(),
        )
 
        return {
            "success": True,
            "attempts": attempt,
            "requirements": requirements,
            "plan": plan,
            "qc": qc,
            "measurement_counts": measurement_counts,
            "image_bytes": image_bytes,
            "metrics": metrics,
            "summary": summary,
        }
 
    raise RuntimeError(
        f"Não foi possível construir um circuito válido após {max_attempts} tentativas. "
        f"Último problema: {last_rejection_reason}"
    )


print(run_quantum_pipeline("crie um circuito Bell com 2 qubits"))  # Expected: Full pipeline result with success=True



