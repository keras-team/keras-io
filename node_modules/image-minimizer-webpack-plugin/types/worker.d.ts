export = worker;
/**
 * @template T
 * @param {import("./index").InternalWorkerOptions<T>} options
 * @returns {Promise<WorkerResult>}
 */
declare function worker<T>(
  options: import("./index").InternalWorkerOptions<T>,
): Promise<WorkerResult>;
declare namespace worker {
  export { WorkerResult, FilenameFn };
}
type WorkerResult = import("./index").WorkerResult;
type FilenameFn = import("./index").FilenameFn;
