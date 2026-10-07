#!/bin/bash
###############################################################################
# GPU Model Monitor - Backend Process (with MQTT Support)
# 
# This script monitors NVIDIA GPU metrics with enhanced process tracking and MQTT publishing.
# Features:
# - Real-time GPU metrics collection
# - Driver and CUDA version tracking
# - Process monitoring with PID, name, and memory usage
# - Process lifetime tracking
# - Historical data management
# - SQLite database for persistence
# - MQTT publishing for Home Assistant integration
# - Multi-GPU support (every GPU is sampled and tracked independently)
###############################################################################

BASE_DIR="/app"
LOG_FILE="$BASE_DIR/gpu_stats.log"
JSON_FILE="$BASE_DIR/gpu_current_stats.json"
HISTORY_DIR="$BASE_DIR/history"
LOG_DIR="$BASE_DIR/logs"
ERROR_LOG="$LOG_DIR/error.log"
WARNING_LOG="$LOG_DIR/warning.log"
DEBUG_LOG="$LOG_DIR/debug.log"
DB_FILE="$HISTORY_DIR/gpu_metrics.db"
MQTT_PUBLISHER="$BASE_DIR/mqtt_publisher.py"
INTERVAL=4  # Time between GPU checks (seconds)
declare -A GPU_MEM_TOTAL     # Total memory in MB per GPU index (updated during monitoring)
declare -A GPU_INDEX_BY_UUID # GPU index lookup by UUID (filled at startup)
GPU_INDEXES=()               # GPU indexes in nvidia-smi order

# Create required directories with proper permissions
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"
mkdir -p "$HISTORY_DIR"
chmod 755 "$HISTORY_DIR"

###############################################################################
# Logging Functions
###############################################################################

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] ERROR: $1" | tee -a "$ERROR_LOG"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] WARNING: $1" | tee -a "$WARNING_LOG"
}

log_debug() {
    if [ "${DEBUG:-}" = "true" ]; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] DEBUG: $1" >> "$DEBUG_LOG"
    fi
}

log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] INFO: $1"
}

###############################################################################
# MQTT Publishing Functions
###############################################################################

# Initialize MQTT publisher (connect and send discovery messages)
mqtt_publisher_init=false

function publish_to_mqtt() {
    local json_file="$1"
    
    # Check if MQTT is enabled
    if [ "${MQTT_ENABLED:-false}" != "true" ]; then
        return 0
    fi
    
    # Check if MQTT publisher exists
    if [ ! -f "$MQTT_PUBLISHER" ]; then
        if [ "$mqtt_publisher_init" = "false" ]; then
            log_warning "MQTT publisher script not found at $MQTT_PUBLISHER"
            mqtt_publisher_init=true
        fi
        return 1
    fi
    
    # Check if Python3 is available
    if ! command -v python3 &> /dev/null; then
        if [ "$mqtt_publisher_init" = "false" ]; then
            log_error "Python3 not found, cannot publish to MQTT"
            mqtt_publisher_init=true
        fi
        return 1
    fi
    
    # Log initialization message once
    if [ "$mqtt_publisher_init" = "false" ]; then
        log_info "MQTT publishing enabled - broker: ${MQTT_HOST:-not_set}:${MQTT_PORT:-1883}"
        mqtt_publisher_init=true
    fi
    
    # Publish metrics to MQTT (run in background to not block monitoring)
    python3 "$MQTT_PUBLISHER" "$json_file" 2>&1 | while read line; do
        log_debug "MQTT: $line"
    done &
}

###############################################################################
# Get GPU list, driver version, and CUDA version
###############################################################################

# Strip leading/trailing whitespace
trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"
    echo "${v%"${v##*[![:space:]]}"}"
}

# Return the argument if it is a number, otherwise 0 (nvidia-smi prints N/A or [N/A])
num_or_zero() {
    local v
    v=$(trim "${1//[\[\]]/}")
    if [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "$v"
    else
        echo "0"
    fi
}

GPU_LIST_JSON="[]"
while IFS=',' read -r idx uuid name; do
    idx=$(trim "$idx"); uuid=$(trim "$uuid"); name=$(trim "$name")
    [ -z "$idx" ] && continue
    GPU_INDEXES+=("$idx")
    GPU_INDEX_BY_UUID["$uuid"]="$idx"
    GPU_LIST_JSON=$(echo "$GPU_LIST_JSON" | jq -c --argjson i "$idx" --arg u "$uuid" --arg n "$name" '. + [{index: $i, uuid: $u, name: $n}]')
done < <(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2>/dev/null)

if [ ${#GPU_INDEXES[@]} -eq 0 ]; then
    GPU_INDEXES=(0)
    GPU_LIST_JSON='[{"index": 0, "uuid": "", "name": "GPU"}]'
fi

DRIVER_VERSION=$(trim "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)")
[ -z "$DRIVER_VERSION" ] && DRIVER_VERSION="Unknown"

# Get CUDA version from nvidia-smi output
CUDA_VERSION_FULL=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
[ -z "$CUDA_VERSION_FULL" ] && CUDA_VERSION_FULL="Unknown"

CONFIG_FILE="$BASE_DIR/gpu_config.json"

# Create config JSON with GPU info (gpu_name kept for the first GPU for older consumers)
jq -n --argjson gpus "$GPU_LIST_JSON" --arg driver "$DRIVER_VERSION" --arg cuda "$CUDA_VERSION_FULL" \
    '{gpu_name: $gpus[0].name, gpu_count: ($gpus | length), gpus: $gpus, driver_version: $driver, cuda_version: $cuda}' > "$CONFIG_FILE"

###############################################################################
# initialize_database: Creates and initializes the SQLite database
###############################################################################
function initialize_database() {
    log_debug "Initializing SQLite database at $DB_FILE"
    
    if [ ! -f "$DB_FILE" ]; then
        log_debug "Creating new database file"
        touch "$DB_FILE"
        chmod 666 "$DB_FILE"
    fi
    
    # Create SQLite tables and indexes
    sqlite3 "$DB_FILE" << SQL
    CREATE TABLE IF NOT EXISTS gpu_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpu_index INTEGER NOT NULL DEFAULT 0,
        timestamp TEXT NOT NULL,
        timestamp_epoch INTEGER NOT NULL,
        temperature REAL NOT NULL,
        utilization REAL NOT NULL,
        memory REAL NOT NULL,
        power REAL NOT NULL
    );
    
    -- Table for tracking processes (one row per GPU + PID)
    CREATE TABLE IF NOT EXISTS gpu_processes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpu_index INTEGER NOT NULL DEFAULT 0,
        pid INTEGER NOT NULL,
        process_name TEXT NOT NULL,
        first_seen INTEGER NOT NULL,
        last_seen INTEGER NOT NULL,
        max_memory REAL NOT NULL,
        avg_memory REAL NOT NULL,
        sample_count INTEGER NOT NULL DEFAULT 1,
        UNIQUE (gpu_index, pid)
    );
    
    -- Table for process snapshots
    CREATE TABLE IF NOT EXISTS process_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpu_index INTEGER NOT NULL DEFAULT 0,
        timestamp_epoch INTEGER NOT NULL,
        pid INTEGER NOT NULL,
        process_name TEXT NOT NULL,
        memory_usage REAL NOT NULL
    );
SQL
    
    if [ $? -ne 0 ]; then
        log_error "Failed to initialize SQLite database"
        return 1
    fi
    
    migrate_database
    
    sqlite3 "$DB_FILE" << SQL
    CREATE INDEX IF NOT EXISTS idx_gpu_metrics_timestamp_epoch ON gpu_metrics(timestamp_epoch);
    CREATE INDEX IF NOT EXISTS idx_gpu_metrics_gpu ON gpu_metrics(gpu_index, timestamp_epoch);
    CREATE INDEX IF NOT EXISTS idx_gpu_processes_pid ON gpu_processes(gpu_index, pid);
    CREATE INDEX IF NOT EXISTS idx_gpu_processes_last_seen ON gpu_processes(last_seen);
    CREATE INDEX IF NOT EXISTS idx_process_snapshots_timestamp ON process_snapshots(timestamp_epoch);
    CREATE INDEX IF NOT EXISTS idx_process_snapshots_pid ON process_snapshots(gpu_index, pid);
SQL
    
    log_debug "Database initialized successfully"
    return 0
}

###############################################################################
# migrate_database: Upgrade a single-GPU database (no gpu_index) in place.
# Existing rows are attributed to GPU 0.
###############################################################################
function migrate_database() {
    local has_col
    has_col=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM pragma_table_info('gpu_processes') WHERE name='gpu_index';")
    [ "$has_col" != "0" ] && return 0
    
    log_info "Migrating database to multi-GPU schema (existing data is assigned to GPU 0)"
    sqlite3 "$DB_FILE" << SQL
    BEGIN;
    ALTER TABLE gpu_metrics ADD COLUMN gpu_index INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE process_snapshots ADD COLUMN gpu_index INTEGER NOT NULL DEFAULT 0;
    -- gpu_processes had UNIQUE(pid); rebuild it with UNIQUE(gpu_index, pid)
    CREATE TABLE gpu_processes_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpu_index INTEGER NOT NULL DEFAULT 0,
        pid INTEGER NOT NULL,
        process_name TEXT NOT NULL,
        first_seen INTEGER NOT NULL,
        last_seen INTEGER NOT NULL,
        max_memory REAL NOT NULL,
        avg_memory REAL NOT NULL,
        sample_count INTEGER NOT NULL DEFAULT 1,
        UNIQUE (gpu_index, pid)
    );
    INSERT INTO gpu_processes_new (pid, process_name, first_seen, last_seen, max_memory, avg_memory, sample_count)
        SELECT pid, process_name, first_seen, last_seen, max_memory, avg_memory, sample_count FROM gpu_processes;
    DROP TABLE gpu_processes;
    ALTER TABLE gpu_processes_new RENAME TO gpu_processes;
    COMMIT;
SQL
    if [ $? -ne 0 ]; then
        log_error "Database migration failed"
        return 1
    fi
}

###############################################################################
# update_process_tracking: Track GPU processes on all GPUs
###############################################################################

# Helper function to validate PID
is_valid_pid() {
    local pid="$1"
    [ -n "$pid" ] && [ "$pid" != "N/A" ] && [ "$pid" != "-" ] && [[ "$pid" =~ ^[0-9]+$ ]]
}

# record_process <gpu_index> <pid> <name> <memory_mb> <timestamp>
# Inserts a snapshot and updates the running per-process statistics.
function record_process() {
    local gpu="$1" pid="$2" name="$3" mem="$4" now="$5"
    
    if ! is_valid_pid "$pid"; then
        log_debug "Invalid PID: $pid, skipping"
        return 1
    fi
    
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        log_debug "Invalid GPU index: $gpu for PID=$pid, skipping"
        return 1
    fi
    
    if ! [[ "$mem" =~ ^[0-9]+$ ]]; then
        log_debug "Invalid memory value: $mem, defaulting to 0"
        mem="0"
    fi
    
    name=$(echo "$name" | sed "s/'/''/g")
    
    if sql_result=$(sqlite3 "$DB_FILE" 2>&1 <<SQL
INSERT INTO process_snapshots (gpu_index, timestamp_epoch, pid, process_name, memory_usage)
VALUES ($gpu, $now, $pid, '$name', $mem);

INSERT INTO gpu_processes (gpu_index, pid, process_name, first_seen, last_seen, max_memory, avg_memory, sample_count)
VALUES ($gpu, $pid, '$name', $now, $now, $mem, $mem, 1)
ON CONFLICT(gpu_index, pid) DO UPDATE SET
    last_seen = $now,
    max_memory = MAX(max_memory, $mem),
    avg_memory = ((avg_memory * sample_count) + $mem) / (sample_count + 1),
    sample_count = sample_count + 1;
SQL
); then
        log_debug "Successfully inserted/updated process GPU=$gpu PID=$pid"
        return 0
    else
        log_error "Failed to insert process GPU=$gpu PID=$pid: $sql_result"
        return 1
    fi
}

function update_process_tracking() {
    local current_time=$(date +%s)
    local process_count=0
    
    # Cache nvidia-smi outputs to avoid repeated calls
    local smi_output=$(nvidia-smi 2>/dev/null)
    # gpu_uuid is included so processes can be matched to the right GPU
    local compute_apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)
    
    log_debug "Starting process tracking at timestamp: $current_time"
    
    # Get current processes using GPU from pmon (captures ALL processes, not just compute apps).
    # pmon reports every GPU; its first column is the GPU index.
    local pmon_raw=$(nvidia-smi pmon -c 1 2>/dev/null)
    local pmon_output=$(echo "$pmon_raw" | grep -v "^#" | awk 'NF')
    
    log_debug "pmon raw output line count: $(echo "$pmon_raw" | wc -l)"
    log_debug "pmon filtered output: $pmon_output"
    
    if [ -z "$pmon_output" ]; then
        log_debug "No GPU processes currently running (pmon empty)"
        
        # Fallback: Try parsing standard nvidia-smi output for processes
        if echo "$smi_output" | grep -q "No running processes found"; then
            log_debug "No GPU processes found (nvidia-smi reports none)"
            return 0
        fi
        
        local process_lines=$(echo "$smi_output" | awk '/Processes:/,/^$/' | grep -E "^\|.*[0-9]+.*MiB.*\|" | grep -v "Processes")
        
        if [ -z "$process_lines" ]; then
            log_debug "No processes found in nvidia-smi table output"
            return 0
        fi
        
        log_debug "Found process lines in nvidia-smi output, parsing..."
        
        while IFS='|' read -r _ content _; do
            [ -z "$content" ] && continue
            
            local gpu=$(echo "$content" | awk '{print $1}')
            local pid=$(echo "$content" | awk '{print $4}')
            local process_name=$(echo "$content" | awk '{if (NF > 6) {for(i=6;i<=NF-1;i++) printf "%s ", $i; printf "\n"} else {print $6}}' | sed 's/[[:space:]]*$//')
            local memory=$(echo "$content" | awk '{print $NF}' | sed 's/MiB//')
            
            pid=$(echo "$pid" | tr -d ' ')
            memory=$(echo "$memory" | tr -d ' ')
            
            log_debug "Parsed nvidia-smi: GPU=$gpu, PID=$pid, Name=$process_name, Mem=$memory"
            
            record_process "$gpu" "$pid" "$process_name" "$memory" "$current_time" && process_count=$((process_count + 1))
        done < <(echo "$process_lines")
        
        log_debug "Processed $process_count processes from nvidia-smi output"
        return 0
    fi
    
    log_debug "Found pmon output, processing..."
    
    while read -r gpu_id pid ptype sm mem_util enc dec command rest; do
        [ -z "$pid" ] && continue
        
        log_debug "Parsed pmon: GPU=$gpu_id, PID=$pid, Type=$ptype, Command=$command"
        
        if ! is_valid_pid "$pid"; then
            log_debug "Invalid PID from pmon: $pid, skipping"
            continue
        fi
        
        # Look the process up among compute apps *on this GPU* (the same PID can run on several GPUs)
        local proc_name="" proc_mem="" app_uuid app_pid app_name app_mem
        while IFS=',' read -r app_uuid app_pid app_name app_mem; do
            app_uuid=$(trim "$app_uuid"); app_pid=$(trim "$app_pid")
            if [ "$app_pid" = "$pid" ] && [ "${GPU_INDEX_BY_UUID[$app_uuid]:-}" = "$gpu_id" ]; then
                proc_name=$(trim "$app_name")
                proc_mem=$(trim "$app_mem")
                break
            fi
        done <<< "$compute_apps"
        
        if [ -n "$proc_name" ]; then
            log_debug "Found in compute_apps: GPU=$gpu_id, PID=$pid, Name=$proc_name, Mem=$proc_mem"
        else
            proc_name="$command"
            proc_mem=$(echo "$smi_output" | grep -E "^\|[[:space:]]+${gpu_id}[[:space:]].*[[:space:]]${pid}[[:space:]].*MiB" | sed 's/.*[[:space:]]\([0-9]\+\)MiB.*/\1/' | head -n1)
            proc_mem="${proc_mem:-0}"
            log_debug "Not in compute_apps, using pmon: GPU=$gpu_id, PID=$pid, Name=$proc_name, Mem=$proc_mem"
        fi
        
        record_process "$gpu_id" "$pid" "$proc_name" "$proc_mem" "$current_time" && process_count=$((process_count + 1))
    done < <(echo "$pmon_output")
    
    log_debug "Processed $process_count processes from pmon output"
}

###############################################################################
# get_current_processes: Get current processes of one GPU as JSON
# Usage: get_current_processes <gpu_index> <total_memory_mb>
###############################################################################
function get_current_processes() {
    local gpu="$1"
    local mem_total="${2:-0}"
    local current_time=$(date +%s)
    local cutoff_time=$((current_time - 10))
    
    log_debug "Getting current processes for GPU $gpu with cutoff_time=$cutoff_time, total_memory=$mem_total"
    
    if result=$(sqlite3 -json "$DB_FILE" 2>&1 <<SQL
    SELECT 
        p.pid,
        p.process_name,
        datetime(p.first_seen, 'unixepoch', 'localtime') as first_seen,
        datetime(p.last_seen, 'unixepoch', 'localtime') as last_seen,
        (p.last_seen - p.first_seen) as lifetime_seconds,
        COALESCE(s.memory_usage, p.max_memory) as memory,
        p.max_memory,
        p.avg_memory,
        p.sample_count
    FROM gpu_processes p
    LEFT JOIN (
        SELECT pid, memory_usage
        FROM process_snapshots
        WHERE gpu_index = $gpu
          AND (pid, timestamp_epoch) IN (
            SELECT pid, MAX(timestamp_epoch)
            FROM process_snapshots
            WHERE gpu_index = $gpu
            GROUP BY pid
        )
    ) s ON p.pid = s.pid
    WHERE p.gpu_index = $gpu
      AND p.last_seen > $cutoff_time
    ORDER BY p.last_seen DESC;
SQL
); then
        log_debug "Current processes query returned: ${result:0:200}..."
        # Enrich with actual OS process start times and memory percentages
        if [ -f "$BASE_DIR/enrich_processes.py" ] && command -v python3 &> /dev/null; then
            # Capture stdout only — enrich_processes.py logs to stderr, and merging it
            # into $result (2>&1) corrupts the JSON when the GPU has no processes.
            result=$(echo "$result" | GPU_MEMORY_TOTAL="$mem_total" python3 "$BASE_DIR/enrich_processes.py" 2>/dev/null)
            if [ $? -ne 0 ]; then
                log_warning "Failed to enrich process data with actual start times"
            fi
        fi
        echo "$result"
    else
        log_error "Failed to query current processes: $result"
        echo "[]"
        return 1
    fi
}

###############################################################################
# safe_write_json: Safely writes JSON data to prevent corruption
###############################################################################
function safe_write_json() {
    local file="$1"
    local content="$2"
    local temp="${file}.tmp"
    local backup="${file}.bak"
    
    echo "$content" > "$temp"
    
    if [ -s "$temp" ]; then
        [ -f "$file" ] && cp "$file" "$backup"
        mv "$temp" "$file"
        [ -f "$backup" ] && rm "$backup"
        return 0
    else
        log_error "Failed to write to temp file: $temp"
        [ -f "$backup" ] && mv "$backup" "$file"
        return 1
    fi
}

###############################################################################
# update_stats: Core function for GPU metrics collection and processing
###############################################################################
update_stats() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local timestamp_epoch=$(date +%s)
    local gpu_stats=$(nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
                     --format=csv,noheader,nounits 2>/dev/null)
    
    if [[ -z "$gpu_stats" ]]; then
        log_error "Failed to get GPU stats output"
        return
    fi
    
    local gpu_objects=()
    local gpu_rows=()
    local idx uuid name temp util mem mem_total power mem_percent
    
    while IFS=',' read -r idx uuid name temp util mem mem_total power; do
        idx=$(trim "$idx")
        [[ "$idx" =~ ^[0-9]+$ ]] || continue
        uuid=$(trim "$uuid")
        name=$(trim "$name")
        temp=$(num_or_zero "$temp")
        util=$(num_or_zero "$util")
        mem=$(num_or_zero "$mem")
        mem_total=$(num_or_zero "$mem_total")
        power=$(num_or_zero "$power")
        
        GPU_MEM_TOTAL[$idx]="$mem_total"
        
        # Calculate memory percentage
        mem_percent=0
        if awk "BEGIN {exit !($mem_total > 0)}"; then
            mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem / $mem_total) * 100}")
        fi
        
        # Insert into database
        sqlite3 "$DB_FILE" <<SQL
        INSERT INTO gpu_metrics (gpu_index, timestamp, timestamp_epoch, temperature, utilization, memory, power)
        VALUES ($idx, '$timestamp', $timestamp_epoch, $temp, $util, $mem, $power);
SQL
        gpu_rows+=("$idx|$uuid|$name|$temp|$util|$mem|$mem_total|$mem_percent|$power")
    done <<< "$gpu_stats"
    
    # Update process tracking for all GPUs once per cycle
    update_process_tracking
    
    local row current_processes
    for row in "${gpu_rows[@]}"; do
        IFS='|' read -r idx uuid name temp util mem mem_total mem_percent power <<< "$row"
        
        # Get current processes for display (pass total memory for percentage calculation)
        current_processes=$(get_current_processes "$idx" "$mem_total")
        
        # Ensure current_processes is valid JSON (empty array if no output)
        if [ -z "$current_processes" ]; then
            current_processes="[]"
        fi
        
        gpu_objects+=("$(jq -n -c \
            --argjson index "$idx" --arg uuid "$uuid" --arg name "$name" \
            --argjson temperature "$temp" --argjson utilization "$util" \
            --argjson memory "$mem" --argjson memory_total "$mem_total" \
            --argjson memory_percent "$mem_percent" --argjson power "$power" \
            --argjson current_processes "$current_processes" \
            '{index: $index, uuid: $uuid, name: $name, temperature: $temperature, utilization: $utilization,
              memory: $memory, memory_total: $memory_total, memory_percent: $memory_percent,
              power: $power, current_processes: $current_processes}')")
    done
    
    # Create JSON content: one entry per GPU
    local json_content
    json_content=$(printf '%s\n' "${gpu_objects[@]}" | jq -s --arg ts "$timestamp" '{timestamp: $ts, gpus: .}')
    
    if [ -z "$json_content" ]; then
        log_error "Failed to build stats JSON"
        return
    fi
    
    # Write JSON safely
    safe_write_json "$JSON_FILE" "$json_content"
    
    # Publish to MQTT if enabled
    publish_to_mqtt "$JSON_FILE"
}

###############################################################################
# export_history_json: Export history to JSON for web display
###############################################################################
function export_history_json() {
    local output_file="$HISTORY_DIR/history.json"
    local cutoff_time=$(( $(date +%s) - 259200 ))  # 3 days
    
    local history_data=$(sqlite3 -json "$DB_FILE" <<SQL
    SELECT 
        gpu_index,
        timestamp,
        temperature,
        utilization,
        memory,
        power
    FROM gpu_metrics
    WHERE timestamp_epoch > $cutoff_time
    ORDER BY timestamp_epoch ASC;
SQL
)
    
    if [ -n "$history_data" ]; then
        echo "$history_data" > "$output_file"
    fi
}

###############################################################################
# export_process_history_json: Export process history to JSON for web display
###############################################################################
function export_process_history_json() {
    local output_file="$HISTORY_DIR/process_history.json"
    
    # Per-GPU total memory, passed to SQL as a lookup table for the percentage calculation
    local totals="" i
    for i in "${GPU_INDEXES[@]}"; do
        totals+="${totals:+,}($i, ${GPU_MEM_TOTAL[$i]:-0})"
    done
    
    local process_history=$(sqlite3 -json "$DB_FILE" <<SQL
    WITH totals(gpu_index, total) AS (VALUES $totals)
    SELECT 
        p.gpu_index,
        p.pid,
        p.process_name,
        datetime(p.first_seen, 'unixepoch', 'localtime') as first_seen,
        datetime(p.last_seen, 'unixepoch', 'localtime') as last_seen,
        (p.last_seen - p.first_seen) as lifetime_seconds,
        p.max_memory,
        p.avg_memory,
        CASE 
            WHEN COALESCE(t.total, 0) > 0 THEN ROUND((p.avg_memory / t.total) * 100, 2)
            ELSE 0 
        END as avg_memory_percent,
        CASE 
            WHEN COALESCE(t.total, 0) > 0 THEN ROUND((p.max_memory / t.total) * 100, 2)
            ELSE 0 
        END as max_memory_percent,
        p.sample_count
    FROM gpu_processes p
    LEFT JOIN totals t ON t.gpu_index = p.gpu_index
    ORDER BY p.last_seen DESC
    LIMIT $(( 100 * ${#GPU_INDEXES[@]} ));
SQL
)
    
    if [ -n "$process_history" ]; then
        echo "$process_history" > "$output_file"
    else
        echo "[]" > "$output_file"
    fi
}

###############################################################################
# cleanup_old_data: Clean up old database records
###############################################################################
function cleanup_old_data() {
    local cutoff_time=$(( $(date +%s) - 259200 ))  # 3 days
    
    sqlite3 "$DB_FILE" <<SQL
    DELETE FROM gpu_metrics WHERE timestamp_epoch < $cutoff_time;
    DELETE FROM process_snapshots WHERE timestamp_epoch < $cutoff_time;
    DELETE FROM gpu_processes WHERE last_seen < $cutoff_time;
    VACUUM;
SQL
}

###############################################################################
# Main execution
###############################################################################

echo "========================================="
echo "GPU Model Monitor (with MQTT)"
echo "========================================="
echo "GPUs: ${#GPU_INDEXES[@]}"
echo "$GPU_LIST_JSON" | jq -r '.[] | "  GPU \(.index): \(.name)"'
echo "Driver: $DRIVER_VERSION"
echo "CUDA: $CUDA_VERSION_FULL"
echo "========================================="

# Initialize database
initialize_database

# Start Python web server in background
python3 "$BASE_DIR/server.py" &
SERVER_PID=$!

# Counter for periodic tasks
export_counter=0
cleanup_counter=0

# Main monitoring loop
while true; do
    update_stats
    
    # Export history every 15 iterations (60 seconds)
    export_counter=$((export_counter + 1))
    if [ $export_counter -ge 15 ]; then
        export_history_json
        export_process_history_json
        export_counter=0
    fi
    
    # Cleanup old data every 900 iterations (1 hour)
    cleanup_counter=$((cleanup_counter + 1))
    if [ $cleanup_counter -ge 900 ]; then
        cleanup_old_data
        cleanup_counter=0
    fi
    
    sleep $INTERVAL
done
