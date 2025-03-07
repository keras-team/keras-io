export type WorkerResult = import("./index").WorkerResult;
export type SquooshOptions = import("./index").SquooshOptions;
export type ImageminOptions = import("imagemin").Options;
export type WebpackError = import("webpack").WebpackError;
export type Module = import("webpack").Module;
export type AssetInfo = import("webpack").AssetInfo;
export type Task<T> = () => Promise<T>;
export type SvgoLib = typeof import("svgo");
export type SvgoOptions = {
  encodeOptions?: Omit<import("svgo").Config, "path" | "datauri"> | undefined;
};
export type SvgoEncodeOptions = Omit<import("svgo").Config, "path" | "datauri">;
export type Uint8ArrayUtf8ByteString = (
  array: number[] | Uint8Array,
  start: number,
  end: number,
) => string;
export type StringToBytes = (string: string) => number[];
export type MetaData = {
  warnings: Array<Error>;
  errors: Array<Error>;
};
export type SharpLib = typeof import("sharp");
export type Sharp = import("sharp").Sharp;
export type ResizeOptions = import("sharp").ResizeOptions & {
  enabled?: boolean;
  unit?: "px" | "percent";
};
export type SharpEncodeOptions = {
  avif?: import("sharp").AvifOptions | undefined;
  gif?: import("sharp").GifOptions | undefined;
  heif?: import("sharp").HeifOptions | undefined;
  jpeg?: import("sharp").JpegOptions | undefined;
  jpg?: import("sharp").JpegOptions | undefined;
  png?: import("sharp").PngOptions | undefined;
  webp?: import("sharp").WebpOptions | undefined;
};
export type SharpFormat = keyof SharpEncodeOptions;
export type SharpOptions = {
  resize?: ResizeOptions | undefined;
  rotate?: number | "auto" | undefined;
  sizeSuffix?: SizeSuffix | undefined;
  encodeOptions?: SharpEncodeOptions | undefined;
};
export type SizeSuffix = (width: number, height: number) => string;
/**
 * Run tasks with limited concurrency.
 * @template T
 * @param {number} limit - Limit of tasks that run at once.
 * @param {Task<T>[]} tasks - List of tasks to run.
 * @returns {Promise<T[]>} A promise that fulfills to an array of the results
 */
export function throttleAll<T>(limit: number, tasks: Task<T>[]): Promise<T[]>;
/**
 * @param {string} url
 * @returns {boolean}
 */
export function isAbsoluteURL(url: string): boolean;
/** @typedef {import("./index").WorkerResult} WorkerResult */
/** @typedef {import("./index").SquooshOptions} SquooshOptions */
/** @typedef {import("imagemin").Options} ImageminOptions */
/** @typedef {import("webpack").WebpackError} WebpackError */
/** @typedef {import("webpack").Module} Module */
/** @typedef {import("webpack").AssetInfo} AssetInfo */
/**
 * @template T
 * @typedef {() => Promise<T>} Task
 */
/**
 * @param {string} filename file path without query params (e.g. `path/img.png`)
 * @param {string} ext new file extension without `.` (e.g. `webp`)
 * @returns {string} new filename `path/img.png` -> `path/img.webp`
 */
export function replaceFileExtension(filename: string, ext: string): string;
/**
 * @template T
 * @param fn {(function(): any) | undefined}
 * @returns {function(): T}
 */
export function memoize<T>(fn: (() => any) | undefined): () => T;
/**
 * @template T
 * @param {ImageminOptions} imageminConfig
 * @returns {Promise<ImageminOptions>}
 */
export function imageminNormalizeConfig<T>(
  imageminConfig: ImageminOptions,
): Promise<ImageminOptions>;
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} options
 * @returns {Promise<WorkerResult | null>}
 */
export function imageminMinify<T>(
  original: WorkerResult,
  options: T,
): Promise<WorkerResult | null>;
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} minimizerOptions
 * @returns {Promise<WorkerResult | null>}
 */
export function imageminGenerate<T>(
  original: WorkerResult,
  minimizerOptions: T,
): Promise<WorkerResult | null>;
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} options
 * @returns {Promise<WorkerResult | null>}
 */
export function squooshMinify<T>(
  original: WorkerResult,
  options: T,
): Promise<WorkerResult | null>;
export namespace squooshMinify {
  export { squooshImagePoolSetup as setup };
  export { squooshImagePoolTeardown as teardown };
}
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} minifyOptions
 * @returns {Promise<WorkerResult | null>}
 */
export function squooshGenerate<T>(
  original: WorkerResult,
  minifyOptions: T,
): Promise<WorkerResult | null>;
export namespace squooshGenerate {
  export { squooshImagePoolSetup as setup };
  export { squooshImagePoolTeardown as teardown };
}
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} minimizerOptions
 * @returns {Promise<WorkerResult | null>}
 */
export function sharpMinify<T>(
  original: WorkerResult,
  minimizerOptions: T,
): Promise<WorkerResult | null>;
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} minimizerOptions
 * @returns {Promise<WorkerResult | null>}
 */
export function sharpGenerate<T>(
  original: WorkerResult,
  minimizerOptions: T,
): Promise<WorkerResult | null>;
/** @typedef {import("svgo")} SvgoLib */
/**
 * @typedef SvgoOptions
 * @type {object}
 * @property {SvgoEncodeOptions} [encodeOptions]
 */
/** @typedef {Omit<import("svgo").Config, "path" | "datauri">} SvgoEncodeOptions */
/**
 * @template T
 * @param {WorkerResult} original
 * @param {T} minimizerOptions
 * @returns {Promise<WorkerResult | null>}
 */
export function svgoMinify<T>(
  original: WorkerResult,
  minimizerOptions: T,
): Promise<WorkerResult | null>;
/** @type {WeakMap<Module, AssetInfo>} */
export const IMAGE_MINIMIZER_PLUGIN_INFO_MAPPINGS: WeakMap<Module, AssetInfo>;
export const ABSOLUTE_URL_REGEX: RegExp;
export const WINDOWS_PATH_REGEX: RegExp;
declare function squooshImagePoolSetup(): void;
declare function squooshImagePoolTeardown(): Promise<void>;
export {};
