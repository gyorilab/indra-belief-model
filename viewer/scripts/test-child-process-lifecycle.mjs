import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createServer } from 'vite';

class FakeChildProcess extends EventEmitter {
	constructor({ exitCode = null, signalCode = null, killReturn = true, throwOnKill = false } = {}) {
		super();
		this.exitCode = exitCode;
		this.signalCode = signalCode;
		this.killReturn = killReturn;
		this.throwOnKill = throwOnKill;
		this.killed = false;
		this.killCalls = 0;
		this.signals = [];
	}

	kill(signal) {
		this.killCalls += 1;
		this.killed = true;
		this.signals.push(signal);
		if (this.throwOnKill) throw new Error('kill failed');
		return this.killReturn;
	}
}

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: false },
	logLevel: 'error'
});

try {
	const { terminateChildProcessWithEscalation } = await server.ssrLoadModule('/src/lib/server/childProcess.ts');

	const stubborn = new FakeChildProcess();
	terminateChildProcessWithEscalation(stubborn, { graceMs: 5 });
	await sleep(50);
	assert.deepEqual(
		stubborn.signals,
		['SIGTERM', 'SIGKILL'],
		'force kill is sent after grace even though child.killed is true after SIGTERM'
	);
	assert.equal(stubborn.listenerCount('exit'), 0, 'exit listener is detached after force escalation');
	assert.equal(stubborn.listenerCount('close'), 0, 'close listener is detached after force escalation');

	const graceful = new FakeChildProcess();
	terminateChildProcessWithEscalation(graceful, { graceMs: 50 });
	graceful.signalCode = 'SIGTERM';
	graceful.emit('exit', null, 'SIGTERM');
	await sleep(60);
	assert.deepEqual(graceful.signals, ['SIGTERM'], 'force kill is skipped after observed exit');
	assert.equal(graceful.listenerCount('exit'), 0, 'exit listener is detached after observed exit');
	assert.equal(graceful.listenerCount('close'), 0, 'close listener is detached after observed exit');

	const closed = new FakeChildProcess();
	terminateChildProcessWithEscalation(closed, { graceMs: 50 });
	closed.emit('close', null, 'SIGTERM');
	await sleep(60);
	assert.deepEqual(closed.signals, ['SIGTERM'], 'force kill is skipped after observed close');
	assert.equal(closed.listenerCount('exit'), 0, 'exit listener is detached after observed close');
	assert.equal(closed.listenerCount('close'), 0, 'close listener is detached after observed close');

	const alreadyExited = new FakeChildProcess({ exitCode: 0 });
	terminateChildProcessWithEscalation(alreadyExited, { graceMs: 1 });
	await sleep(5);
	assert.deepEqual(alreadyExited.signals, [], 'already-exited child is not signaled');

	const missing = new FakeChildProcess({ killReturn: false });
	terminateChildProcessWithEscalation(missing, { graceMs: 1 });
	await sleep(5);
	assert.deepEqual(missing.signals, ['SIGTERM'], 'missing process is not force-killed after failed SIGTERM delivery');

	const throwing = new FakeChildProcess({ throwOnKill: true });
	terminateChildProcessWithEscalation(throwing, { graceMs: 1 });
	await sleep(5);
	assert.equal(throwing.killCalls, 1, 'kill exceptions stop escalation after one attempt');
	assert.equal(throwing.listenerCount('exit'), 0, 'exit listener is detached after kill exception');
	assert.equal(throwing.listenerCount('close'), 0, 'close listener is detached after kill exception');

	console.log('child process lifecycle tests passed');
} finally {
	await server.close();
}
