export = loader;
/**
 * @template T
 * @this {import("webpack").LoaderContext<LoaderOptions<T>>}
 * @param {Buffer} content
 * @returns {Promise<Buffer | undefined>}
 */
declare function loader<T>(
  this: import("webpack").LoaderContext<LoaderOptions<T>>,
  content: Buffer,
): Promise<Buffer | undefined>;
declare namespace loader {
  export {
    raw,
    Schema,
    Compilation,
    WorkerResult,
    Minimizer,
    Generator,
    LoaderOptions,
  };
}
declare var raw: boolean;
type Schema = import("schema-utils/declarations/validate").Schema;
type Compilation = import("webpack").Compilation;
type WorkerResult = import("./utils").WorkerResult;
/**
 * <T>
 */
type Minimizer<T> = import("./index").Minimizer<T>;
/**
 * <T>
 */
type Generator<T> = import("./index").Generator<T>;
/**
 * <T>
 */
type LoaderOptions<T> = {
  /**
   * Allows to choose how errors are displayed.
   */
  severityError?: string | undefined;
  minimizer?:
    | import("./index").Minimizer<T>
    | import("./index").Minimizer<T>[]
    | undefined;
  generator?: import("./index").Generator<T>[] | undefined;
};
