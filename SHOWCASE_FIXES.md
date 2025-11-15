# MAPF-GNN Showcase Repository Fixes

## Focus: ML Correctness + Clean Engineering for Portfolio

### 🔴 Priority 1: Critical Math/ML Fixes (MUST DO)

- [ ] **Fix bias addition bug** in `models/networks/gnn.py` lines 97 & 206
  - Changes `node_feats += node_feats + self.b` → `node_feats += self.b`
- [ ] **Fix in-place adjacency matrix modification** in `models/networks/gnn.py` line 63
  - Don't modify input adjacency matrix
- [ ] **Fix inefficient loss computation** in `train.py` lines 71-74
  - Use proper PyTorch loss aggregation
- [ ] **Add gradient clipping** to prevent training instabilities
  - Essential for GNN stability

### 🔴 Priority 2: Make It Runnable (MUST DO)

- [ ] **Remove ALL hardcoded paths**
  - `train.py` lines 3-4: Remove absolute paths
  - Use argparse for config file selection
  - Use os.path.join() or pathlib everywhere
- [ ] **Fix cross-platform compatibility**
  - Replace backslashes with forward slashes in configs
  - Test on Linux/Mac

### 🟡 Priority 3: Essential Engineering (SHOULD DO)

- [ ] **Add core docstrings** to main classes/functions
  - Network classes
  - Training functions
  - Key algorithms (CBS, GNN layers)
- [ ] **Add type hints** to main functions
  - Focus on public interfaces
- [ ] **Create clean requirements.txt**
  - Only essential packages with versions
- [ ] **Add random seed setting** for reproducibility
- [ ] **Add basic logging** instead of print statements
- [ ] **Save training metrics** (loss curves, success rates)

### 🟡 Priority 4: Documentation (SHOULD DO)

- [ ] **Update README.md** with:
  - Clear installation instructions
  - How to train the model
  - How to run the example
  - Architecture overview
  - Key results/metrics
  - Paper citation
- [ ] **Add example notebook** showing trained model performance
- [ ] **Document config parameters**

### 🟢 Priority 5: Nice to Have (IF TIME)

- [ ] **Add learning rate scheduling**
- [ ] **Implement model checkpointing**
- [ ] **Add simple unit tests** for GNN forward pass
- [ ] **Profile and document performance**
- [ ] **Add ablation study results**

## Implementation Order

1. **Fix math bugs** (15 mins)
2. **Remove hardcoded paths** (30 mins)
3. **Add argparse and cross-platform support** (45 mins)
4. **Fix loss and add gradient clipping** (20 mins)
5. **Add seeds and basic logging** (30 mins)
6. **Clean requirements.txt** (10 mins)
7. **Add essential docstrings** (1 hour)
8. **Update README** (45 mins)
9. **Test everything works** (30 mins)

**Total Time Estimate: 4-5 hours**

## Success Criteria for Showcase

✅ Math is correct (no bugs in algorithms)
✅ Code runs on any machine (no hardcoded paths)
✅ Clear instructions in README
✅ Training reproduces claimed results
✅ Clean, readable code with basic documentation
✅ Professional git history

## What We're NOT Doing

❌ Full test suite
❌ Production deployment features
❌ API development
❌ Extensive refactoring
❌ Performance optimization beyond basics
❌ Comprehensive error handling everywhere

---
Let's start with the critical fixes!