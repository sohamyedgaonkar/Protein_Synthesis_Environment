# Protein Folding Environment Evaluation Results

---

## 🧪 TASK_1

### Initial State
- **Energy:** 7.000  
- **Score:** 0.270  
- **Hydrophobic Contacts:** 18  
- **Collisions:** 5  

---

### Step Logs

**Step 1**  
- Action: `rotate_psi`  
- Residue: `3`  
- Delta: `60.0`  
- Reward: `+0.812`  
- Energy: `-7.000`  
- Score: `0.647`  
- Contacts: `17`  
- Collisions: `2`  

⚠️ Environment signalled done. Stopping early.

---

### Final Summary
- **Steps Taken:** 1  
- **Final Energy:** -7.000  
- **Final Score:** 0.647  
- **Final Contacts:** 17  
- **Final Collisions:** 2  

---

## 🧪 TASK_2

### Initial State
- **Energy:** 7.000  
- **Score:** 0.270  
- **Hydrophobic Contacts:** 18  
- **Collisions:** 5  

---

### Step Logs

**Step 1**  
- Action: `rotate_psi`  
- Residue: `2`  
- Delta: `30.0`  
- Reward: `+0.730`  
- Energy: `1.000`  
- Score: `0.579`  
- Contacts: `19`  
- Collisions: `4`  

**Step 2**  
- Action: `pivot_rotation`  
- Residue: `12`  
- Delta: `-60.0`  
- Reward: `+0.596`  
- Energy: `-2.000`  
- Score: `0.662`  
- Contacts: `22`  
- Collisions: `4`  

**Step 3**  
- Action: `rotate_phi`  
- Residue: `6`  
- Delta: `30.0`  
- Reward: `+0.875`  
- Energy: `-12.000`  
- Score: `0.706`  
- Contacts: `22`  
- Collisions: `2`  

---

### Final Summary
- **Steps Taken:** 3  
- **Final Energy:** -12.000  
- **Final Score:** 0.706  
- **Final Contacts:** 22  
- **Final Collisions:** 2  

---

## 🧪 TASK_3

### Initial State
- **Energy:** 7.000  
- **Score:** 0.218  
- **Hydrophobic Contacts:** 23  
- **Collisions:** 6  

---

### Step Logs

**Step 1**  
- Action: `rotate_psi`  
- Residue: `3`  
- Delta: `60.0`  
- Reward: `+0.763`  
- Energy: `-11.000`  
- Score: `0.600`  
- Contacts: `21`  
- Collisions: `2`  

**Step 2**  
- Action: `rotate_phi`  
- Residue: `1`  
- Delta: `-60.0`  
- Reward: `+0.674`  
- Energy: `-16.000`  
- Score: `0.656`  
- Contacts: `21`  
- Collisions: `1`  

**Step 3**  
- Action: `rotate_psi`  
- Residue: `10`  
- Delta: `30.0`  
- Reward: `+0.439`  
- Energy: `-17.000`  
- Score: `0.793`  
- Contacts: `17`  
- Collisions: `0`  

**Step 4**  
- Action: `rotate_psi`  
- Residue: `16`  
- Delta: `-60.0`  
- Reward: `+0.512`  
- Energy: `-19.000`  
- Score: `0.807`  
- Contacts: `19`  
- Collisions: `0`  

**Step 5**  
- Action: `segment_flip`  
- Segment: `12 → 16`  
- Reward: `+0.528`  
- Energy: `-21.000`  
- Score: `0.822`  
- Contacts: `21`  
- Collisions: `0`  

---

### Final Summary
- **Steps Taken:** 5  
- **Final Energy:** -21.000  
- **Final Score:** 0.822  
- **Final Contacts:** 21  
- **Final Collisions:** 0  

---

## ✅ Overall Observation

- The agent consistently **reduces energy effectively**
- **Hydrophobic clustering improves over time**
- **Collisions reduce to zero in later tasks**
- Best performance observed in **TASK_3 (Score: 0.822)**

---