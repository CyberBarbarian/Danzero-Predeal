# DanZero Agent Logging Improvements

## Overview
This document describes the improvements made to the logging system for DanZero agents (ai1-ai6) to reduce verbose output and implement efficient log management.

## Changes Made

### 1. Centralized Logging Configuration
- **File**: `guandan/agent/logging_config.py`
- **Purpose**: Provides a centralized logging system for all agents
- **Features**:
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR, DISABLE)
  - Environment variable control (`DANZERO_LOG_LEVEL`)
  - Per-agent logger instances
  - Structured log formatting

### 2. Agent Modifications
Modified all agent client files to use the new logging system:
- `guandan/agent/baselines/rule/ai1/client.py`
- `guandan/agent/baselines/rule/ai2/client.py`
- `guandan/agent/baselines/rule/ai3/client.py`
- `guandan/agent/baselines/rule/ai4/client.py`
- `guandan/agent/baselines/rule/ai6/client.py`
- `guandan/agent/baselines/rule/ai6/action.py`

**Changes**:
- Replaced `print()` statements with appropriate logging levels
- Added logger instances to each agent class
- Converted verbose debug prints to `logger.debug()` calls
- Converted error prints to `logger.error()` calls
- Converted info prints to `logger.info()` calls

### 3. Log Level Strategy
- **Default Level**: WARNING (minimal output)
- **Debug Information**: Moved to DEBUG level (hidden by default)
- **Error Messages**: Kept at ERROR level (always visible)
- **Agent Lifecycle**: INFO level for important events

## Usage

### Environment Variable Control
```bash
export DANZERO_LOG_LEVEL=WARNING  # Default - minimal output
export DANZERO_LOG_LEVEL=INFO     # Show info and above
export DANZERO_LOG_LEVEL=DEBUG    # Show all messages (verbose)
export DANZERO_LOG_LEVEL=ERROR    # Show only errors
export DANZERO_LOG_LEVEL=DISABLE  # Disable all logging
```

### Programmatic Control
```python
from guandan.agent.logging_config import AgentLogger
import logging

# Set default level for all agents
AgentLogger.set_default_level(logging.WARNING)  # Minimal
AgentLogger.set_default_level(logging.INFO)     # Normal
AgentLogger.set_default_level(logging.DEBUG)    # Verbose
AgentLogger.disable_logging()                   # Silent

# Set level for specific agent
AgentLogger.set_agent_level("ai6", logging.DEBUG)
```

## Benefits

### Before (Old System)
- ❌ Console flooded with debug information
- ❌ No control over log verbosity
- ❌ Always verbose output
- ❌ Difficult to identify important messages
- ❌ Poor performance due to excessive I/O

### After (New System)
- ✅ Clean output by default
- ✅ Configurable log levels
- ✅ Structured, timestamped logs
- ✅ Easy to identify log sources
- ✅ Better performance (minimal I/O by default)
- ✅ Environment variable control
- ✅ Per-agent logging control

## Example Output

### Default (WARNING level)
```
14:04:58 - danzero.agent.ai1 - WARNING - Agent 1 closed down: code=1000, reason=Normal closure
14:04:58 - danzero.agent.ai1 - ERROR - Error in ai1 agent 1: KeyError 'actionList'
```

### Verbose (DEBUG level)
```
14:04:58 - danzero.agent.ai6 - DEBUG - 可选动作范围为：0至5
14:04:58 - danzero.agent.ai6 - DEBUG - 这里会反映每个msg: {'stage': 'play'...}
14:04:58 - danzero.agent.ai6 - DEBUG - 我先手的msg: {'stage': 'play'...}
14:04:58 - danzero.agent.ai6 - INFO - Agent 6 selected action index: 3
```

## Migration Notes
- All existing agent functionality is preserved
- No breaking changes to agent interfaces
- Logging is backward compatible
- Agents can be used exactly as before, but with better log control

## Files Modified
1. `guandan/agent/logging_config.py` (new)
2. `guandan/agent/baselines/rule/ai1/client.py`
3. `guandan/agent/baselines/rule/ai2/client.py`
4. `guandan/agent/baselines/rule/ai3/client.py`
5. `guandan/agent/baselines/rule/ai4/client.py`
6. `guandan/agent/baselines/rule/ai6/client.py`
7. `guandan/agent/baselines/rule/ai6/action.py`

## Testing
The logging system has been tested and verified to work correctly with all agents. All agents can be created and used without issues, and the logging levels can be controlled as expected.
