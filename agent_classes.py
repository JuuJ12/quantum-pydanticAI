from enum import Enum #serve para criar um tipo de dado enumerado, ou seja, um tipo de dado que pode assumir um conjunto finito de valores, nesse caso, os tipos de circuitos que o agente construtor pode criar
from pydantic import BaseModel, Field #basemodel serve para criar modelos de dados, ou seja, estruturas de dados que podem ser validadas e convertidas para outros formatos, e field serve para definir os campos do modelo, ou seja, os atributos que o modelo vai ter
from typing import Optional, List, Literal


class CircuitType(str, Enum): #serve para definir os tipos de circuitos que o agente construtor pode criar, ou seja, circuitos computacionais, circuitos de Bell, circuitos de GHZ e circuitos de superposição
    COMPUTATIONAL = "computational"
    BELL = "bell"
    GHZ = "ghz"
    SUPERPOSITION = "superposition"


class StructuredCircuit(BaseModel): # estrutura de dados para requisição do circuito, ou seja, o usuário vai pedir e vai ficar nesse formato
    objective: str = Field(description="The objective of the circuit")
    num_qubits: int = Field(description="Number of qubits in the circuit")
    target_state: str = Field(description="Desired final quantum state")
    circuit_type: CircuitType = Field(
        default=CircuitType.COMPUTATIONAL,
        description="Type of target circuit/state: computational, bell, ghz or superposition",
    )


class Gate(BaseModel): # estrutura de dados para a construção do circuito. Para cada qubit ele vai aplicar uma porta, e essa é a estrutura de dados para cada porta, ou seja, o nome da porta, os qubits alvo e os qubits de controle (se houver)
    gate_name: Literal["h", "x", "cx", "rz"] = Field(description="Name of the quantum gate (h, x, cx, rz)")
    target_qubits: List[int] = Field(description="List of target qubits for the gate")
    control_qubits: Optional[int | List[int]] = Field(
        default=None,
        description="Control qubit index (or list) if needed",
    )
    theta: Optional[float] = Field(
        default=None,
        description="Rotation angle in radians used only when gate_name is rz",
    )


class CircuitPlan(BaseModel): # estrutura da lista de portas que o agente construtor vai retornar, ou seja, a lista de portas que ele vai aplicar no circuito para atingir o objetivo do usuário
    gates: List[Gate] = Field(
        description="List of quantum gates to be applied in the circuit"
    )

class CircuitMetrics(BaseModel):
    fidelity: float = Field(description="Fidelity of the executed circuit compared to the target state")
    depth: int = Field(description="Depth of the executed circuit")
    gate_count: int = Field(description="Total number of gates used in the executed circuit")


class VerificationResult(BaseModel):
    approved: bool
    reason: Optional[str] = None


class IsQuantumAwnser(BaseModel):
    is_quantum: bool
