// Package jobs implements a tiny in-memory background job manager. Each job
// runs in its own goroutine, appends log lines that clients can tail via
// Server-Sent Events, and finishes in Succeeded or Failed state with an
// arbitrary JSON-serializable Result.
package jobs

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

type Status string

const (
	StatusPending   Status = "pending"
	StatusRunning   Status = "running"
	StatusSucceeded Status = "succeeded"
	StatusFailed    Status = "failed"
)

// Job represents one background unit of work (a download or a merge).
type Job struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Status    Status      `json:"status"`
	CreatedAt time.Time   `json:"createdAt"`
	UpdatedAt time.Time   `json:"updatedAt"`
	Error     string      `json:"error,omitempty"`
	Result    interface{} `json:"result,omitempty"`

	mu       sync.Mutex
	log      []string
	subs     map[chan string]struct{}
	finished chan struct{}
}

// Manager owns all jobs created during the process lifetime.
type Manager struct {
	mu   sync.RWMutex
	jobs map[string]*Job
}

// NewManager creates an empty job manager.
func NewManager() *Manager {
	return &Manager{jobs: map[string]*Job{}}
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// Create registers a new job of the given type and returns it. The caller is
// expected to launch a goroutine that eventually calls Succeed or Fail.
func (m *Manager) Create(jobType string) *Job {
	j := &Job{
		ID:        newID(),
		Type:      jobType,
		Status:    StatusPending,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		subs:      map[chan string]struct{}{},
		finished:  make(chan struct{}),
	}
	m.mu.Lock()
	m.jobs[j.ID] = j
	m.mu.Unlock()
	return j
}

// Get looks up a job by id.
func (m *Manager) Get(id string) (*Job, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	j, ok := m.jobs[id]
	return j, ok
}

// ActiveCount returns how many not-yet-finished (pending or running) jobs of
// the given type currently exist. Used by the API layer to cap concurrent
// downloads/merges - see CAVEAT.md's "resource-exhaustion abuse" mitigation.
func (m *Manager) ActiveCount(jobType string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	n := 0
	for _, j := range m.jobs {
		if j.Type != jobType {
			continue
		}
		status, _ := j.Snapshot()
		if status == StatusPending || status == StatusRunning {
			n++
		}
	}
	return n
}

// List returns all known jobs, most recent first.
func (m *Manager) List() []*Job {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]*Job, 0, len(m.jobs))
	for _, j := range m.jobs {
		out = append(out, j)
	}
	return out
}

// MarkRunning transitions a job to the running state.
func (j *Job) MarkRunning() {
	j.mu.Lock()
	j.Status = StatusRunning
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

// Log appends a line to the job's log and fans it out to any live subscribers.
func (j *Job) Log(format string, args ...interface{}) {
	line := fmt.Sprintf(format, args...)
	j.mu.Lock()
	j.log = append(j.log, line)
	j.UpdatedAt = time.Now()
	subs := make([]chan string, 0, len(j.subs))
	for ch := range j.subs {
		subs = append(subs, ch)
	}
	j.mu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- line:
		default:
			// slow subscriber, drop the line rather than block the job
		}
	}
}

// Succeed marks the job as successfully completed with the given result.
func (j *Job) Succeed(result interface{}) {
	j.mu.Lock()
	j.Status = StatusSucceeded
	j.Result = result
	j.UpdatedAt = time.Now()
	close(j.finished)
	j.mu.Unlock()
}

// Fail marks the job as failed with the given error.
func (j *Job) Fail(err error) {
	j.mu.Lock()
	j.Status = StatusFailed
	j.Error = err.Error()
	j.UpdatedAt = time.Now()
	close(j.finished)
	j.mu.Unlock()
}

// Snapshot returns the current log lines and job status, safe to call
// concurrently with Log/Succeed/Fail.
func (j *Job) Snapshot() (status Status, logLines []string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	cp := make([]string, len(j.log))
	copy(cp, j.log)
	return j.Status, cp
}

// Subscribe registers a channel that receives future log lines. It returns
// an unsubscribe function that must be called when the caller is done.
func (j *Job) Subscribe() (ch chan string, unsubscribe func()) {
	ch = make(chan string, 64)
	j.mu.Lock()
	j.subs[ch] = struct{}{}
	j.mu.Unlock()
	return ch, func() {
		j.mu.Lock()
		delete(j.subs, ch)
		j.mu.Unlock()
	}
}

// Done returns a channel that is closed when the job finishes (success or failure).
func (j *Job) Done() <-chan struct{} {
	return j.finished
}
