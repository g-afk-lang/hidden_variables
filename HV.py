import cv2
import numpy as np
from scipy.stats import pearsonr
import time
from random import randint

import requests
bet = input("INPUT: ")
# -------- Windice request helpers (single call safe) --------
API_URL = "https://windice.io/api/v1/api/roll"
API_HEADERS = {
    "Authorization": "",
    "Content-Type": "application/json",
}

def roll_low_once():
    data = {"curr": "win", "bet": 1, "game": "in", "low": 1, "high": 5000}
    try:
        r = requests.post(API_URL, headers=API_HEADERS, json=data, timeout=10)
        r.raise_for_status()
        js = r.json()
        print(js)
        bet = js['data']['result']
        # Robust extract: return numeric “result” if present, else None
        return js.get("data", {}).get("result", None)
    except Exception as e:
        print(f"Windice low roll error: {e}")
        return None

def roll_high_once():
    data = {"curr": "win", "bet": 1, "game": "in", "low": 5000, "high": 9999}
    try:
        r = requests.post(API_URL, headers=API_HEADERS, json=data, timeout=10)
        r.raise_for_status()
        js = r.json()
        print(js)
        bet = js['data']['result']
        return js.get("data", {}).get("result", None)
    except Exception as e:
        print(f"Windice high roll error: {e}")
        return None
        
        
# FIXED IMPORTS for Qiskit 1.0+
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
    print("🔬 Qiskit quantum computing library loaded!")
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Qiskit not available. Install with: pip install qiskit qiskit-aer")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn not available. Install with: pip install scikit-learn")

class HiddenVariablePredictor:
    """Predicts hidden deterministic forces from pixel patterns"""
    
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.linear_model = LinearRegression()
            self.nonlinear_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.training_data = []
        self.hidden_variables = []
        self.trained = False
        self.frame_history = []
        
    def extract_pixel_features(self, frame):
        """Extract deterministic features from pixel patterns"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Feature extraction: various statistical measures
        features = []
        
        # 1. Block intensity variations (hidden force patterns)
        h, w = gray.shape
        for block_size in [2, 4, 8]:
            for i in range(0, h-block_size, block_size):
                for j in range(0, w-block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    features.extend([
                        np.mean(block),           # Average intensity
                        np.std(block),            # Variation (hidden force strength)
                        np.min(block),            # Minimum (force floor)
                        np.max(block) - np.min(block)  # Dynamic range
                    ])
        
        # 2. Gradient-based hidden forces
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),     # Overall force magnitude
            np.std(gradient_magnitude),      # Force variation
            np.max(gradient_magnitude),      # Peak force
            np.sum(gradient_magnitude > np.mean(gradient_magnitude))  # Active force regions
        ])
        
        # 3. Temporal coherence forces (if we have history)
        if len(self.frame_history) > 0:
            prev_gray = self.frame_history[-1]
            frame_diff = cv2.absdiff(gray, prev_gray)
            features.extend([
                np.mean(frame_diff),         # Temporal force
                np.std(frame_diff),          # Temporal instability
                np.max(frame_diff)           # Maximum change force
            ])
        else:
            features.extend([0, 0, 0])  # No temporal data yet
        
        # 4. Fourier-based hidden periodicities
        f_transform = np.fft.fft2(gray)
        f_magnitude = np.abs(f_transform)
        features.extend([
            np.mean(f_magnitude),        # Average frequency force
            np.std(f_magnitude),         # Frequency variation
            np.max(f_magnitude)          # Dominant frequency force
        ])
        
        return np.array(features)
    
    def train_predictor(self, features, hidden_var):
        """Train the hidden variable predictor"""
        self.training_data.append(features)
        self.hidden_variables.append(hidden_var)
        
        # Train after collecting enough data
        if len(self.training_data) >= 10 and SKLEARN_AVAILABLE:
            X = np.array(self.training_data)
            y = np.array(self.hidden_variables)
            
            # Train both linear and nonlinear models
            self.linear_model.fit(X, y)
            self.nonlinear_model.fit(X, y)
            self.trained = True
            
            print(f"   🧠 Hidden variable predictor trained on {len(self.training_data)} samples")
            
            # Clear old training data to prevent memory issues
            if len(self.training_data) > 50:
                self.training_data = self.training_data[-25:]
                self.hidden_variables = self.hidden_variables[-25:]
    
    def predict_hidden_variable(self, features):
        """Predict hidden variable from pixel features"""
        if not self.trained or not SKLEARN_AVAILABLE:
            return 0
        
        features = features.reshape(1, -1)
        
        # Ensemble prediction: combine linear and nonlinear models
        linear_pred = self.linear_model.predict(features)[0]
        nonlinear_pred = self.nonlinear_model.predict(features)[0]
        
        # Weighted combination
        ensemble_pred = 0.3 * linear_pred + 0.7 * nonlinear_pred
        
        return ensemble_pred

class RealTimeQuantumCameraProcessor:
    def __init__(self):
        self.simulator = AerSimulator() if QISKIT_AVAILABLE else None
        self.quantum_frames_processed = 0
        self.real_time_quantum_states = []
        self.hidden_predictor = HiddenVariablePredictor()
        
    def quantum_process_camera_frame(self, frame):
        if not QISKIT_AVAILABLE:
            return frame, None, 0, 0, 0
        
        try:
            # Extract hidden deterministic forces from pixels
            pixel_features = self.hidden_predictor.extract_pixel_features(frame)
            
            # Generate the "true" hidden variable (what we're trying to predict)
            true_hidden_var = randint(1,100)
            
            # Train the predictor
            self.hidden_predictor.train_predictor(pixel_features, bet)
            
            # Predict hidden variable
            predicted_hidden_var = self.hidden_predictor.predict_hidden_variable(pixel_features)
            
            print(f"   🔮 Hidden variable - True: {true_hidden_var:.4f}, Predicted: {predicted_hidden_var:.4f}")
            if true_hidden_var < 50:
                roll_low_once()
                return
            if true_hidden_var > 50:
                roll_high_once()
                return
            # Process frame quantumly with hidden variable influence
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            small_frame = cv2.resize(gray_frame, (4, 4))
            normalized_pixels = small_frame.flatten().astype(np.float32) / 255.0
            
            # MODIFY quantum angles based on predicted hidden variable
            hidden_influence = predicted_hidden_var * 0.1  # Scale influence
            quantum_angles = (normalized_pixels * np.pi) + hidden_influence
            
            n_qubits = len(quantum_angles)
            quantum_circuit = QuantumCircuit(n_qubits)
            
            print(f"   🔬 Quantum circuit with {n_qubits} qubits + hidden force influence")
            
            # Encode pixels with hidden variable modulation
            for i, angle in enumerate(quantum_angles):
                quantum_circuit.ry(angle, i)
            
            # Add entanglement influenced by hidden variable
            entanglement_strength = int(abs(predicted_hidden_var) % 4) + 1
            for i in range(n_qubits - 1):
                if i % 4 != 3 and i % entanglement_strength == 0:
                    quantum_circuit.cx(i, i + 1)
                if i < n_qubits - 4 and i % entanglement_strength == 0:
                    quantum_circuit.cx(i, i + 4)
            
            # Superposition influenced by hidden variable
            superposition_pattern = int(abs(predicted_hidden_var * 10) % 4) + 1
            for i in range(0, n_qubits, superposition_pattern):
                quantum_circuit.h(i)
            
            # Add phase rotations based on hidden variable
            phase_rotation = predicted_hidden_var * 0.05
            for i in range(n_qubits):
                if i % 2 == 0:
                    quantum_circuit.rz(phase_rotation, i)
            
            # Execute quantum circuit
            statevector = Statevector.from_instruction(quantum_circuit)
            quantum_amplitudes = statevector.data
            
            self.real_time_quantum_states.append(quantum_amplitudes)
            self.quantum_frames_processed += 1
            
            # Convert back to image
            quantum_intensities = np.abs(quantum_amplitudes[:16]) ** 2
            quantum_frame_small = (quantum_intensities.reshape(4, 4) * 255).astype(np.uint8)
            quantum_frame = cv2.resize(quantum_frame_small, gray_frame.shape[::-1])
            
            # Calculate quantum coherence influenced by hidden variable
            base_coherence = np.abs(np.sum(quantum_amplitudes * quantum_amplitudes.conj()))
            hidden_coherence = base_coherence * (1 + predicted_hidden_var * 0.1)
            
            print(f"   ⚛️  Quantum processed with hidden forces: coherence = {hidden_coherence:.4f}")
            
            # Store frame history for temporal analysis
            self.hidden_predictor.frame_history.append(gray_frame)
            if len(self.hidden_predictor.frame_history) > 5:
                self.hidden_predictor.frame_history.pop(0)
            
            return quantum_frame, quantum_amplitudes, hidden_coherence, true_hidden_var, predicted_hidden_var
            
        except Exception as e:
            print(f"   ❌ Quantum processing error: {e}")
            return frame, None, 0, 0, 0

class RealTimeQuantumCameraAnalyzer:
    def __init__(self):
        self.quantum_processor = RealTimeQuantumCameraProcessor()
        self.frame_count = 0
        self.previous_quantum_state = None
        self.quantum_violations_detected = 0
        self.total_quantum_correlations = []
        self.hidden_variable_accuracy = []
        
        if QISKIT_AVAILABLE:
            print("🔬 REAL-TIME QUANTUM CAMERA ANALYZER with HIDDEN VARIABLES")
            print("⚛️  Processing live camera feed with quantum circuits")
            print("🔮 Predicting hidden deterministic forces from pixel patterns")
        else:
            print("❌ Quantum libraries required!")
    
    def process_camera_stream(self):
        # Use DirectShow backend to avoid MSMF errors
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("❌ Camera not accessible with DirectShow")
            print("   Try closing other applications using the camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("\n🚀 STARTING QUANTUM CAMERA WITH HIDDEN VARIABLE PREDICTION")
        print("📹 Using DirectShow backend for camera access")
        print("🔮 Extracting hidden deterministic forces from pixels")
        print("⚛️  Quantum processing influenced by predicted hidden variables")
        print("\nPress 'q' to quit")
        
        try:

            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame - camera may be in use")
            
            print(f"\n📹 Frame {self.frame_count}:")
            
            # QUANTUM PROCESS with HIDDEN VARIABLE PREDICTION
            quantum_frame, current_quantum_state, coherence, true_hidden, predicted_hidden = self.quantum_processor.quantum_process_camera_frame(frame)
            
            # Track prediction accuracy
            if true_hidden != 0:
                accuracy = 1 - abs(true_hidden - predicted_hidden) / (abs(true_hidden) + 1e-6)
                self.hidden_variable_accuracy.append(accuracy)
            
            # QUANTUM CORRELATION with previous frame
            if self.previous_quantum_state is not None and current_quantum_state is not None:
                fidelity = np.abs(np.vdot(current_quantum_state, self.previous_quantum_state))**2
                print(f"   🔗 Quantum frame correlation: fidelity={fidelity:.4f}")
                self.total_quantum_correlations.append(fidelity)
            
            # Display results
            try:
                display_frame = cv2.resize(frame, (320, 240))
                quantum_display = cv2.resize(quantum_frame, (320, 240))
                if len(quantum_display.shape) == 2:
                    quantum_display = cv2.cvtColor(quantum_display, cv2.COLOR_GRAY2BGR)
                
                combined_display = np.hstack([display_frame, quantum_display])
                
                # Add information overlay
                cv2.putText(combined_display, "Original", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined_display, "Quantum + Hidden Forces", (330, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(combined_display, f"Q-Frames: {self.quantum_processor.quantum_frames_processed}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(combined_display, f"Hidden Var: {predicted_hidden:.3f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.putText(combined_display, f"Coherence: {coherence:.3f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                if len(self.hidden_variable_accuracy) > 0:
                    avg_accuracy = np.mean(self.hidden_variable_accuracy[-10:])
                    cv2.putText(combined_display, f"Prediction Acc: {avg_accuracy:.3f}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Color border based on hidden variable strength
                if abs(predicted_hidden) > 1.0:
                    border_color = (0, 0, 255)  # Red for strong hidden forces
                elif abs(predicted_hidden) > 0.5:
                    border_color = (0, 255, 255)  # Yellow for moderate forces
                else:
                    border_color = (0, 255, 0)  # Green for weak forces
                
                cv2.rectangle(combined_display, (0, 0), (combined_display.shape[1]-1, combined_display.shape[0]-1), border_color, 3)
                
                cv2.imshow('Quantum Camera with Hidden Variable Prediction', combined_display)
                
                    
            except cv2.error as e:
                print(f"   ⚠️ Display error: {e}")
                pass
            
            self.previous_quantum_state = current_quantum_state
            self.frame_count += 1
            
            # Print summary every 10 frames
            if self.frame_count % 1 == 0:
                avg_accuracy = np.mean(self.hidden_variable_accuracy[-10:]) if self.hidden_variable_accuracy else 0
                print(f"\n📊 Hidden Variable Prediction Summary:")
                print(f"   • Frames processed: {self.quantum_processor.quantum_frames_processed}")
                print(f"   • Average prediction accuracy: {avg_accuracy:.4f}")
                print(f"   • Hidden variable predictor trained: {self.quantum_processor.hidden_predictor.trained}")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping quantum camera processing...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🔬 FINAL ANALYSIS:")
            print(f"   • Total frames quantumly processed: {self.quantum_processor.quantum_frames_processed}")
            if self.hidden_variable_accuracy:
                print(f"   • Overall hidden variable prediction accuracy: {np.mean(self.hidden_variable_accuracy):.4f}")
            print(f"   • Hidden variable predictor trained: {self.quantum_processor.hidden_predictor.trained}")

if __name__ == "__main__":
    print("🔬 QUANTUM CAMERA with HIDDEN DETERMINISTIC FORCES")
    print("📹 Live camera → Hidden variable extraction → Quantum processing")
    print("🔮 Predicting hidden forces that influence quantum behavior")
    
    if not QISKIT_AVAILABLE:
        print("\n❌ INSTALL QUANTUM COMPUTING LIBRARY:")
        print("pip install qiskit qiskit-aer")
        exit(1)
    
    if not SKLEARN_AVAILABLE:
        print("⚠️  For enhanced hidden variable prediction, install:")
        print("pip install scikit-learn")
    while True:
        try:
            analyzer = RealTimeQuantumCameraAnalyzer()
            analyzer.process_camera_stream()
            
        except Exception as e:
            print(f"❌ Quantum camera processing error: {e}")