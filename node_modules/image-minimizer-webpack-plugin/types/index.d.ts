export = ImageMinimizerPlugin;
/**
 * @template T, [G=T]
 * @extends {WebpackPluginInstance}
 */
declare class ImageMinimizerPlugin<T, G = T> {
  /**
   * @param {PluginOptions<T, G>} [options={}] Plugin options.
   */
  constructor(options?: PluginOptions<T, G> | undefined);
  /**
   * @private
   */
  private options;
  /**
   * @private
   * @param {Compiler} compiler
   * @param {Compilation} compilation
   * @param {Record<string, Source>} assets
   * @returns {Promise<void>}
   */
  private optimize;
  /**
   * @private
   */
  private setupAll;
  /**
   * @private
   */
  private teardownAll;
  /**
   * @param {import("webpack").Compiler} compiler
   */
  apply(compiler: import("webpack").Compiler): void;
}
declare namespace ImageMinimizerPlugin {
  export {
    loader,
    imageminNormalizeConfig,
    imageminMinify,
    imageminGenerate,
    squooshMinify,
    squooshGenerate,
    sharpMinify,
    sharpGenerate,
    svgoMinify,
    Schema,
    WebpackPluginInstance,
    Compiler,
    Compilation,
    WebpackError,
    Asset,
    AssetInfo,
    Source,
    Module,
    ImageminMinifyFunction,
    SquooshMinifyFunction,
    Rule,
    Rules,
    FilterFn,
    ImageminOptions,
    SquooshOptions,
    WorkerResult,
    Task,
    CustomOptions,
    InferDefaultType,
    BasicTransformerOptions,
    ResizeOptions,
    BasicTransformerImplementation,
    BasicTransformerHelpers,
    TransformerFunction,
    PathData,
    FilenameFn,
    Transformer,
    Minimizer,
    Generator,
    InternalWorkerOptions,
    InternalLoaderOptions,
    PluginOptions,
  };
}
declare var loader: string;
import { imageminNormalizeConfig } from "./utils.js";
import { imageminMinify } from "./utils.js";
import { imageminGenerate } from "./utils.js";
import { squooshMinify } from "./utils.js";
import { squooshGenerate } from "./utils.js";
import { sharpMinify } from "./utils.js";
import { sharpGenerate } from "./utils.js";
import { svgoMinify } from "./utils.js";
type Schema = import("schema-utils/declarations/validate").Schema;
type WebpackPluginInstance = import("webpack").WebpackPluginInstance;
type Compiler = import("webpack").Compiler;
type Compilation = import("webpack").Compilation;
type WebpackError = import("webpack").WebpackError;
type Asset = import("webpack").Asset;
type AssetInfo = import("webpack").AssetInfo;
type Source = import("webpack").sources.Source;
type Module = import("webpack").Module;
type ImageminMinifyFunction = typeof imageminMinify;
type SquooshMinifyFunction = typeof squooshMinify;
type Rule = RegExp | string;
type Rules = Rule[] | Rule;
type FilterFn = (source: Buffer, sourcePath: string) => boolean;
type ImageminOptions = {
  plugins: Array<
    string | [string, Record<string, any>?] | import("imagemin").Plugin
  >;
};
type SquooshOptions = {
  [x: string]: any;
};
type WorkerResult = {
  filename: string;
  data: Buffer;
  warnings: Array<Error>;
  errors: Array<Error>;
  info: AssetInfo;
};
type Task<T> = {
  name: string;
  info: AssetInfo;
  inputSource: Source;
  output:
    | (WorkerResult & {
        source?: Source;
      })
    | undefined;
  cacheItem: ReturnType<ReturnType<Compilation["getCache"]>["getItemCache"]>;
  transformer: Transformer<T> | Transformer<T>[];
};
type CustomOptions = {
  [key: string]: any;
};
type InferDefaultType<T> = T extends infer U ? U : CustomOptions;
type BasicTransformerOptions<T> = InferDefaultType<T> | undefined;
type ResizeOptions = {
  width?: number | undefined;
  height?: number | undefined;
  unit?: "px" | "percent" | undefined;
  enabled?: boolean | undefined;
};
type BasicTransformerImplementation<T> = (
  original: WorkerResult,
  options?: BasicTransformerOptions<T>,
) => Promise<WorkerResult | null>;
type BasicTransformerHelpers = {
  setup?: (() => void) | undefined;
  teardown?: (() => void) | undefined;
};
type TransformerFunction<T> = BasicTransformerImplementation<T> &
  BasicTransformerHelpers;
type PathData = {
  filename?: string | undefined;
};
type FilenameFn = (
  pathData: PathData,
  assetInfo?: import("webpack").AssetInfo | undefined,
) => string;
type Transformer<T> = {
  implementation: TransformerFunction<T>;
  options?: BasicTransformerOptions<T>;
  filter?: FilterFn | undefined;
  filename?: string | FilenameFn | undefined;
  preset?: string | undefined;
  type?: "import" | "asset" | undefined;
};
type Minimizer<T> = Omit<Transformer<T>, "preset" | "type">;
type Generator<T> = Transformer<T>;
type InternalWorkerOptions<T> = {
  filename: string;
  info?: AssetInfo | undefined;
  input: Buffer;
  transformer: Transformer<T> | Transformer<T>[];
  severityError?: string | undefined;
  generateFilename?: Function | undefined;
};
type InternalLoaderOptions<T> = import("./loader").LoaderOptions<T>;
type PluginOptions<T, G> = {
  /**
   * Test to match files against.
   */
  test?: Rule | undefined;
  /**
   * Files to include.
   */
  include?: Rule | undefined;
  /**
   * Files to exclude.
   */
  exclude?: Rule | undefined;
  /**
   * Allows to setup the minimizer.
   */
  minimizer?:
    | (T extends any[]
        ? { [P in keyof T]: Minimizer<T[P]> }
        : Minimizer<T> | Minimizer<T>[])
    | undefined;
  /**
   * Allows to set the generator.
   */
  generator?:
    | (G extends any[]
        ? { [P_1 in keyof G]: Generator<G[P_1]> }
        : Generator<G>[])
    | undefined;
  /**
   * Automatically adding `imagemin-loader`.
   */
  loader?: boolean | undefined;
  /**
   * Maximum number of concurrency optimization processes in one time.
   */
  concurrency?: number | undefined;
  /**
   * Allows to choose how errors are displayed.
   */
  severityError?: string | undefined;
  /**
   * Allows to remove original assets. Useful for converting to a `webp` and remove original assets.
   */
  deleteOriginalAssets?: boolean | undefined;
};
