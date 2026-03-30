#!/bin/bash
# Parse pytest summary lines from saved CI logs and aggregate results.
# Matches the GitHub Checks UI by mapping errors→failed and xfailed→skipped.

LOGDIR="${1:-ci-logs}"

total_passed=0
total_failed=0
total_skipped=0
jobs_parsed=0
jobs_missing=0

# Track totals excluding jobs with missing/manual logs
ui_passed=0
ui_failed=0
ui_skipped=0
ui_jobs=0
manual_jobs=""

printf "%-15s %8s %8s %8s  %s\n" "JOB_ID" "PASSED" "FAILED" "SKIPPED" "NOTES"
printf "%-15s %8s %8s %8s  %s\n" "------" "------" "------" "-------" "-----"

for logfile in "$LOGDIR"/*.log; do
    job_id=$(basename "$logfile" .log)
    is_manual=false

    # Detect logs that were manually pulled (no CI step prefix, starts with "log not found")
    if head -1 "$logfile" | grep -q "log not found"; then
        is_manual=true
    fi

    # Extract the pytest summary line. Two formats:
    #   = 6 failed, 17 passed, 2 skipped, ... in 1367.55s (0:22:47) =
    #   ============= 5 passed, 1 skipped, ... 1 error in 13.24s ==============
    summary=$(grep -oP '=+ \d+ \w+,.* in \d+\.\d+s.*=+' "$logfile" | tail -1)

    if [ -z "$summary" ]; then
        printf "%-15s %8s %8s %8s  %s\n" "$job_id" "-" "-" "-" "NO SUMMARY FOUND"
        jobs_missing=$((jobs_missing + 1))
        continue
    fi

    passed=$(echo "$summary" | grep -oP '\d+ passed' | grep -oP '\d+' || echo 0)
    failed=$(echo "$summary" | grep -oP '\d+ failed' | grep -oP '\d+' || echo 0)
    skipped=$(echo "$summary" | grep -oP '\d+ skipped' | grep -oP '\d+' || echo 0)
    errors=$(echo "$summary" | grep -oP '\d+ error' | grep -oP '\d+' || echo 0)
    xfailed=$(echo "$summary" | grep -oP '\d+ xfailed' | grep -oP '\d+' || echo 0)

    : "${passed:=0}" "${failed:=0}" "${skipped:=0}" "${errors:=0}" "${xfailed:=0}"

    # Match GitHub Checks UI: errors count as failed, xfailed count as skipped
    failed=$((failed + errors))
    skipped=$((skipped + xfailed))

    notes=""
    [ "$errors" -gt 0 ] && notes="${notes}${errors}err "
    [ "$xfailed" -gt 0 ] && notes="${notes}${xfailed}xfail "
    if $is_manual; then
        notes="${notes}(manual log)"
        manual_jobs="${manual_jobs} ${job_id}"
    fi

    printf "%-15s %8s %8s %8s  %s\n" "$job_id" "$passed" "$failed" "$skipped" "$notes"

    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))
    total_skipped=$((total_skipped + skipped))
    jobs_parsed=$((jobs_parsed + 1))

    if ! $is_manual; then
        ui_passed=$((ui_passed + passed))
        ui_failed=$((ui_failed + failed))
        ui_skipped=$((ui_skipped + skipped))
        ui_jobs=$((ui_jobs + 1))
    fi
done

printf "%-15s %8s %8s %8s\n" "------" "------" "------" "-------"
printf "%-15s %8d %8d %8d\n" "TOTAL" "$total_passed" "$total_failed" "$total_skipped"
total_ran=$((total_passed + total_failed + total_skipped))
echo ""
echo "All $jobs_parsed jobs: $total_ran tests ($total_passed passed, $total_failed failed, $total_skipped skipped)"

if [ -n "$manual_jobs" ]; then
    ui_ran=$((ui_passed + ui_failed + ui_skipped))
    echo ""
    echo "Excluding manual logs ($manual_jobs):"
    echo "  $ui_jobs jobs: $ui_ran tests ($ui_passed passed, $ui_failed failed, $ui_skipped skipped)"
fi

if [ "$jobs_missing" -gt 0 ]; then
    echo ""
    echo "WARNING: $jobs_missing job(s) had no pytest summary found"
fi
