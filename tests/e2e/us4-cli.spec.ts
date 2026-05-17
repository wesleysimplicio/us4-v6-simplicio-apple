import {expect, test, type TestInfo} from "@playwright/test";
import {execFile} from "node:child_process";
import {existsSync} from "node:fs";
import path from "node:path";
import {promisify} from "node:util";

const execFileAsync = promisify(execFile);
const repoRoot = path.resolve(__dirname, "..", "..");
const cliPath = path.join(repoRoot, "bin", "cli.js");
const packageJson = require(path.join(repoRoot, "package.json")) as {
  version: string;
};

type CliRun = {
  stdout: string; stderr : string;
};

async function runCli(args: string[]): Promise<CliRun> {
  const {stdout, stderr} = await execFileAsync(
      process.execPath,
      [ cliPath, ...args ],
      {
        cwd : repoRoot,
        env : {
          ...process.env,
          NO_COLOR : "1",
        },
      },
  );

  return {
    stdout : stdout.trim(),
    stderr : stderr.trim(),
  };
}

async function attachCommand(
    testInfo: TestInfo,
    label: string,
    result: CliRun,
    ): Promise<void> {
  await testInfo.attach(`stdout-${label}`, {
    body : result.stdout || "(empty)",
    contentType : "text/plain",
  });
  await testInfo.attach(`stderr-${label}`, {
    body : result.stderr || "(empty)",
    contentType : "text/plain",
  });
}

test.describe("Starter CLI smoke", () => {
  test("exposes version contract in text and JSON", async ({}, testInfo) => {
    const textResult = await runCli([ "--version" ]);
    await attachCommand(testInfo, "version-text", textResult);

    expect(textResult.stderr).toBe("");
    expect(textResult.stdout).toBe(packageJson.version);

    const jsonResult = await runCli([ "--version", "--json" ]);
    await attachCommand(testInfo, "version-json", jsonResult);

    expect(jsonResult.stderr).toBe("");
    expect(JSON.parse(jsonResult.stdout)).toMatchObject({
      cli : "us4-cli",
      version : packageJson.version,
    });
  });

  test("exposes probe contract in text and JSON", async ({}, testInfo) => {
    const textResult = await runCli([ "--probe" ]);
    await attachCommand(testInfo, "probe-text", textResult);

    expect(textResult.stderr).toBe("");
    expect(textResult.stdout).toContain(`us4-cli ${packageJson.version}`);
    expect(textResult.stdout).toContain("mode: ");
    expect(textResult.stdout).toContain("platform: ");
    expect(textResult.stdout).toContain("memory: ");

    const jsonResult = await runCli([ "--probe", "--json" ]);
    await attachCommand(testInfo, "probe-json", jsonResult);

    expect(jsonResult.stderr).toBe("");
    expect(JSON.parse(jsonResult.stdout)).toMatchObject({
      cli : "us4-cli",
      version : packageJson.version,
      probe : {
        platform : expect.any(String),
        arch : expect.any(String),
        cpuModel : expect.any(String),
        logicalCores : expect.any(Number),
        totalMemoryBytes : expect.any(Number),
        totalMemoryGiB : expect.any(Number),
        appleSilicon : expect.any(Boolean),
        aneEligible : expect.any(Boolean),
      },
      mode : {
        requested : "auto",
        selected : expect.any(String),
        taxonomy : expect.arrayContaining([ "FULL", "MICRO_PLUS", "NANO" ]),
        source : "memory-tier",
      },
    });
  });

  test("keeps mode auto JSON aligned with the probe shape",
       async ({}, testInfo) => {
         const modeResult = await runCli([ "--mode", "auto", "--json" ]);
         await attachCommand(testInfo, "mode-auto-json", modeResult);

         expect(modeResult.stderr).toBe("");
         expect(JSON.parse(modeResult.stdout)).toMatchObject({
           cli : "us4-cli",
           version : packageJson.version,
           probe : {
             platform : expect.any(String),
             arch : expect.any(String),
           },
           mode : {
             requested : "auto",
             selected : expect.any(String),
             taxonomy : [
               "FULL",
               "BALANCED_PLUS",
               "DEGRADED",
               "ULTRA_LOW",
               "MICRO",
               "MICRO_PLUS",
               "NANO",
             ],
             source : "memory-tier",
           },
         });
       });
});

test.describe("Native CLI sprint 02 contract", () => {
  const nativeCliCandidates = [
    path.join(repoRoot, "build", "apps", "us4-cli.exe"),
    path.join(repoRoot, "build", "apps", "us4-cli"),
    path.join(repoRoot, "build", "us4-cli"),
    path.join(repoRoot, "build", "us4-cli.exe"),
    path.join(repoRoot, "build", "Release", "us4-cli.exe"),
  ];
  const nativeCliPath = nativeCliCandidates.find(
      (candidate) => existsSync(candidate),
  );

  test.skip(!nativeCliPath, "native us4-cli is not available in this host");

  test("probe command exposes native acceleration profile",
       async ({}, testInfo) => {
         const {stdout, stderr} = await execFileAsync(
             nativeCliPath!,
             [ "--probe", "--json" ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );

         await testInfo.attach("stdout-native-probe", {
           body : stdout.trim() || "(empty)",
           contentType : "text/plain",
         });
         await testInfo.attach("stderr-native-probe", {
           body : stderr.trim() || "(empty)",
           contentType : "text/plain",
         });

         expect(stderr.trim()).toBe("");
         expect(JSON.parse(stdout)).toMatchObject({
           version : expect.any(String),
           platform : expect.any(String),
           architecture : expect.any(String),
           chip : expect.any(String),
           has_mlx : expect.any(Boolean),
           has_metal : expect.any(Boolean),
           has_neon : expect.any(Boolean),
           neon_vector_bits : expect.any(Number),
           has_performance_cores : expect.any(Boolean),
           has_efficiency_cores : expect.any(Boolean),
           metal_device : expect.any(String),
           metal_queue_label : expect.any(String),
           metal_threads_per_group : expect.any(Number),
           supports_unified_memory : expect.any(Boolean),
           metal_init_stage : expect.any(String),
           metal_queue_created : expect.any(Boolean),
           metal_autorelease_boundary_requested : expect.any(Boolean),
           metal_objc_boundary_supported : expect.any(Boolean),
           metal_reason : expect.any(String),
           recommended_mode : expect.any(String),
         });
       });

  test("run command emits generated scalar tokens", async ({}, testInfo) => {
    const fixturePath = path.join(
        repoRoot,
        "tests",
        "fixtures",
        "models",
        "qwen-0.5b",
        "model.us4manifest",
    );
    const {stdout, stderr} = await execFileAsync(
        nativeCliPath!,
        [
          "run",
          "--model",
          "qwen-0.5b",
          "--model-path",
          fixturePath,
          "--backend",
          "metal",
          "--prompt",
          "hi",
          "--max-tokens",
          "5",
          "--json",
        ],
        {
          cwd : repoRoot,
          env : {
            ...process.env,
            NO_COLOR : "1",
          },
        },
    );

    await testInfo.attach("stdout-native-run", {
      body : stdout.trim() || "(empty)",
      contentType : "text/plain",
    });
    await testInfo.attach("stderr-native-run", {
      body : stderr.trim() || "(empty)",
      contentType : "text/plain",
    });

    expect(stderr.trim()).toBe("");
    expect(JSON.parse(stdout)).toMatchObject({
      model : "qwen-0.5b-fixture",
      asset_format : "fixture-manifest",
      backend : "scalar",
      backend_reason : "requested-backend-unavailable",
      fallback : true,
      shared_allocations : 0,
      metal_dispatches : 0,
      mlx_operation_count : 0,
      kv_cache_hit : false,
      kv_restored_from_cold_store : false,
      kv_page_count : 1,
      kv_hot_pages : 1,
      kv_warm_pages : 0,
      kv_cold_pages : 0,
      kv_summary_rows : 0,
      prefix_cache_entries : 1,
      mlx_plan_built : false,
      mlx_evaluated : false,
      metal_device : expect.any(String),
      metal_queue_label : expect.any(String),
      asset_path : expect.any(String),
      prompt_tokens : [ "hi" ],
      generated_tokens : expect.any(Array),
    });
    expect(JSON.parse(stdout).generated_tokens.length)
        .toBeGreaterThanOrEqual(
            5,
        );
  });

  test("list-models exposes available native adapters",
       async ({}, testInfo) => {
         const {stdout, stderr} = await execFileAsync(
             nativeCliPath!,
             [ "list-models", "--json" ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );

         await testInfo.attach("stdout-native-list-models", {
           body : stdout.trim() || "(empty)",
           contentType : "text/plain",
         });
         await testInfo.attach("stderr-native-list-models", {
           body : stderr.trim() || "(empty)",
           contentType : "text/plain",
         });

         expect(stderr.trim()).toBe("");
         const payload = JSON.parse(stdout) as {
           models: Array<Record<string, unknown>>;
         };
         expect(Array.isArray(payload.models)).toBeTruthy();

         const qwen = payload.models.find((model) => model.family === "qwen");
         const llama = payload.models.find((model) => model.family === "llama");
         const deepseek = payload.models.find(
             (model) => model.family === "deepseek",
         );

         expect(qwen).toMatchObject({
           family : "qwen",
           model : "qwen-0.5b",
           architecture : "dense",
           minimum_mode : "NANO",
           supports_moe : false,
           supports_mlx : false,
           supports_metal : true,
           supports_prompt_run : true,
           preferred_backend : expect.any(String),
           preferred_backend_reason : expect.any(String),
           preferred_mode : expect.any(String),
         });
         expect(llama).toMatchObject({
           family : "llama",
           model : "llama-3.1-8b",
           architecture : "dense",
           supports_mlx : true,
           supports_metal : true,
           preferred_backend : expect.any(String),
         });
         expect(deepseek).toMatchObject({
           family : "deepseek",
           model : "deepseek-v2-lite",
           architecture : "moe",
           supports_moe : true,
           supports_mlx : true,
           supports_metal : true,
           preferred_backend : expect.any(String),
         });
       });

  test("run accepts explicit scalar backend without fallback",
       async ({}, testInfo) => {
         const fixturePath = path.join(
             repoRoot,
             "tests",
             "fixtures",
             "models",
             "qwen-0.5b",
             "model.us4manifest",
         );
         const {stdout, stderr} = await execFileAsync(
             nativeCliPath!,
             [
               "run",
               "--model-path",
               fixturePath,
               "--backend",
               "scalar",
               "--prompt",
               "hi",
               "--max-tokens",
               "4",
               "--json",
             ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );

         await testInfo.attach("stdout-native-scalar", {
           body : stdout.trim() || "(empty)",
           contentType : "text/plain",
         });
         await testInfo.attach("stderr-native-scalar", {
           body : stderr.trim() || "(empty)",
           contentType : "text/plain",
         });

         expect(stderr.trim()).toBe("");
         expect(JSON.parse(stdout)).toMatchObject({
           model : "qwen-0.5b-fixture",
           backend : "scalar",
           backend_reason : "requested",
           fallback : false,
           shared_allocations : 0,
           metal_dispatches : 0,
           mlx_operation_count : 0,
           kv_cache_hit : false,
           kv_restored_from_cold_store : false,
           kv_page_count : 1,
           kv_summary_rows : 0,
         });
       });

  test("run honors explicit neon backend semantics", async ({}, testInfo) => {
    const fixturePath = path.join(
        repoRoot,
        "tests",
        "fixtures",
        "models",
        "qwen-0.5b",
        "model.us4manifest",
    );

    const probeResult = await execFileAsync(
        nativeCliPath!,
        [ "--probe", "--json" ],
        {
          cwd : repoRoot,
          env : {
            ...process.env,
            NO_COLOR : "1",
          },
        },
    );
    const probe = JSON.parse(probeResult.stdout) as { has_neon: boolean; };

    const {stdout, stderr} = await execFileAsync(
        nativeCliPath!,
        [
          "run",
          "--model-path",
          fixturePath,
          "--backend",
          "neon",
          "--prompt",
          "hi",
          "--max-tokens",
          "4",
          "--json",
        ],
        {
          cwd : repoRoot,
          env : {
            ...process.env,
            NO_COLOR : "1",
          },
        },
    );

    await testInfo.attach("stdout-native-neon", {
      body : stdout.trim() || "(empty)",
      contentType : "text/plain",
    });
    await testInfo.attach("stderr-native-neon", {
      body : stderr.trim() || "(empty)",
      contentType : "text/plain",
    });

    expect(stderr.trim()).toBe("");
    const payload = JSON.parse(stdout) as Record<string, unknown>;
    expect(Array.isArray(payload.generated_tokens)).toBeTruthy();
    expect(
        (payload.generated_tokens as unknown[]).length,
        )
        .toBeGreaterThanOrEqual(4);

    if (probe.has_neon) {
      expect(payload).toMatchObject({
        model : "qwen-0.5b-fixture",
        backend : "neon",
        backend_reason : "requested",
        fallback : false,
      });
    } else {
      expect(payload).toMatchObject({
        model : "qwen-0.5b-fixture",
        backend : "scalar",
        backend_reason : "requested-backend-unavailable",
        fallback : true,
      });
    }
  });

  test("run can resolve model from manifest without explicit --model",
       async ({}, testInfo) => {
         const fixturePath = path.join(
             repoRoot,
             "tests",
             "fixtures",
             "models",
             "qwen-0.5b",
             "model.us4manifest",
         );
         const {stdout, stderr} = await execFileAsync(
             nativeCliPath!,
             [
               "run",
               "--model-path",
               fixturePath,
               "--prompt",
               "hi",
               "--max-tokens",
               "4",
               "--json",
             ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );

         await testInfo.attach("stdout-native-auto", {
           body : stdout.trim() || "(empty)",
           contentType : "text/plain",
         });
         await testInfo.attach("stderr-native-auto", {
           body : stderr.trim() || "(empty)",
           contentType : "text/plain",
         });

         expect(stderr.trim()).toBe("");
         expect(JSON.parse(stdout)).toMatchObject({
           model : "qwen-0.5b-fixture",
           backend_reason : expect.stringMatching(/^auto-/),
           fallback : false,
           shared_allocations : 0,
           metal_dispatches : 0,
           mlx_operation_count : 0,
           kv_cache_hit : false,
           kv_restored_from_cold_store : false,
           kv_page_count : 1,
           kv_summary_rows : 0,
         });
       });

  test("run rejects invalid backend values", async ({}, testInfo) => {
    const fixturePath = path.join(
        repoRoot,
        "tests",
        "fixtures",
        "models",
        "qwen-0.5b",
        "model.us4manifest",
    );
    let failure: {stdout: string; stderr : string}|undefined;

    try {
      await execFileAsync(
          nativeCliPath!,
          [
            "run",
            "--model-path",
            fixturePath,
            "--backend",
            "nope",
            "--prompt",
            "hi",
            "--json",
          ],
          {
            cwd : repoRoot,
            env : {
              ...process.env,
              NO_COLOR : "1",
            },
          },
      );
    } catch (error) {
      failure = {
        stdout : String((error as {stdout?: string}).stdout ?? "").trim(),
        stderr : String((error as {stderr?: string}).stderr ?? "").trim(),
      };
    }

    await testInfo.attach("stderr-native-invalid-backend", {
      body : failure?.stderr || "(empty)",
      contentType : "text/plain",
    });

    expect(failure).toBeTruthy();
    expect(failure?.stderr).toContain("Invalid --backend value");
  });

  test("llama manifest path exposes host-aware backend semantics",
       async ({}, testInfo) => {
         const fixturePath = path.join(
             repoRoot,
             "tests",
             "fixtures",
             "models",
             "llama-3.1-8b",
         );
         const probeResult = await execFileAsync(
             nativeCliPath!,
             [ "--probe", "--json" ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );
         const probe = JSON.parse(probeResult.stdout) as {
           has_metal: boolean;
           has_neon: boolean;
         };
         const {stdout, stderr} = await execFileAsync(
             nativeCliPath!,
             [
               "run",
               "--model-path",
               fixturePath,
               "--backend",
               "metal",
               "--prompt",
               "hello",
               "--max-tokens",
               "4",
               "--json",
             ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );

         await testInfo.attach("stdout-native-llama", {
           body : stdout.trim() || "(empty)",
           contentType : "text/plain",
         });
         await testInfo.attach("stderr-native-llama", {
           body : stderr.trim() || "(empty)",
           contentType : "text/plain",
         });

         expect(stderr.trim()).toBe("");
         const payload = JSON.parse(stdout) as Record<string, unknown>;
         expect(payload).toMatchObject({
           family : "llama",
           model : "llama-3.1-8b-fixture",
           asset_format : "fixture-manifest",
           asset_path : expect.stringContaining("llama-3.1-8b"),
           prompt_tokens : [ "hello" ],
           generated_tokens : expect.any(Array),
         });
         expect((payload.generated_tokens as unknown[]).length)
             .toBeGreaterThanOrEqual(
                 4,
             );

         if (probe.has_metal) {
           expect(payload).toMatchObject({
             backend : "metal",
             backend_reason : "requested",
             fallback : false,
           });
           expect(Number(payload.shared_allocations)).toBeGreaterThan(0);
           expect(Number(payload.metal_dispatches)).toBeGreaterThan(0);
           expect(payload.metal_queue_label).not.toBe("disabled");
         } else {
           expect(payload).toMatchObject({
             backend : probe.has_neon ? "neon" : "scalar",
             backend_reason : "requested-backend-unavailable",
             fallback : true,
             shared_allocations : 0,
             metal_dispatches : 0,
           });
         }
       });

  test("llama gguf loader contract stays visible in native cli",
       async ({}, testInfo) => {
         const ggufPath = path.join(
             repoRoot,
             "tests",
             "fixtures",
             "models",
             "llama-3.1-8b",
             "toy-llama.gguf",
         );
         const {stdout, stderr} = await execFileAsync(
             nativeCliPath!,
             [
               "run",
               "--model-path",
               ggufPath,
               "--prompt",
               "",
               "--max-tokens",
               "4",
               "--json",
             ],
             {
               cwd : repoRoot,
               env : {
                 ...process.env,
                 NO_COLOR : "1",
               },
             },
         );

         await testInfo.attach("stdout-native-llama-gguf", {
           body : stdout.trim() || "(empty)",
           contentType : "text/plain",
         });
         await testInfo.attach("stderr-native-llama-gguf", {
           body : stderr.trim() || "(empty)",
           contentType : "text/plain",
         });

         expect(stderr.trim()).toBe("");
         const payload = JSON.parse(stdout) as Record<string, unknown>;
         expect(payload).toMatchObject({
           family : "llama",
           model : "toy-llama",
           asset_format : "gguf",
           asset_path : expect.stringContaining("toy-llama.gguf"),
           backend_reason : expect.stringMatching(/^auto-/),
           fallback : false,
           prompt_tokens : [ "hello" ],
           generated_tokens : expect.any(Array),
         });
         expect((payload.generated_tokens as unknown[]).length)
             .toBeGreaterThanOrEqual(
                 4,
             );
       });

  test("deepseek moe path emits moe family output", async ({}, testInfo) => {
    const {stdout, stderr} = await execFileAsync(
        nativeCliPath!,
        [
          "run",
          "--model",
          "deepseek-v2-lite",
          "--prompt",
          "hi",
          "--max-tokens",
          "4",
          "--json",
        ],
        {
          cwd : repoRoot,
          env : {
            ...process.env,
            NO_COLOR : "1",
          },
        },
    );

    await testInfo.attach("stdout-native-deepseek", {
      body : stdout.trim() || "(empty)",
      contentType : "text/plain",
    });
    await testInfo.attach("stderr-native-deepseek", {
      body : stderr.trim() || "(empty)",
      contentType : "text/plain",
    });

    expect(stderr.trim()).toBe("");
    expect(JSON.parse(stdout)).toMatchObject({
      family : "deepseek",
      backend : "scalar",
      shared_allocations : 0,
      metal_dispatches : 0,
      mlx_operation_count : 0,
      kv_cache_hit : false,
      kv_restored_from_cold_store : false,
      kv_page_count : 1,
      kv_summary_rows : 0,
      generated_tokens : expect.any(Array),
    });
  });

  test(
      "bitnet gguf loader keeps low-bit telemetry visible without explicit model",
      async ({}, testInfo) => {
        const ggufPath = path.join(
            repoRoot,
            "tests",
            "fixtures",
            "models",
            "bitnet-b1.58-2b",
            "toy-bitnet.gguf",
        );
        const {stdout, stderr} = await execFileAsync(
            nativeCliPath!,
            [
              "run",
              "--model-path",
              ggufPath,
              "--prompt",
              "",
              "--max-tokens",
              "4",
              "--json",
            ],
            {
              cwd : repoRoot,
              env : {
                ...process.env,
                NO_COLOR : "1",
              },
            },
        );

        await testInfo.attach("stdout-native-bitnet-gguf", {
          body : stdout.trim() || "(empty)",
          contentType : "text/plain",
        });
        await testInfo.attach("stderr-native-bitnet-gguf", {
          body : stderr.trim() || "(empty)",
          contentType : "text/plain",
        });

        expect(stderr.trim()).toBe("");
        const payload = JSON.parse(stdout) as Record<string, unknown>;
        expect(payload).toMatchObject({
          family : "bitnet",
          model : "toy-bitnet",
          asset_format : "gguf",
          asset_path : expect.stringContaining("toy-bitnet.gguf"),
          prompt_tokens : [ "hi" ],
          weight_dtype : "int8",
          dequant_path : "groupwise-int8",
          generated_tokens : expect.any(Array),
        });
        expect((payload.generated_tokens as unknown[]).length)
            .toBeGreaterThanOrEqual(
                4,
            );
      });

  test(
      "ternary safetensors loader keeps low-bit telemetry visible without explicit model",
      async ({}, testInfo) => {
        const tensorPath = path.join(
            repoRoot,
            "tests",
            "fixtures",
            "models",
            "pt-bitnet-ternary-2b",
            "toy-ternary.safetensors",
        );
        const {stdout, stderr} = await execFileAsync(
            nativeCliPath!,
            [
              "run",
              "--model-path",
              tensorPath,
              "--prompt",
              "",
              "--max-tokens",
              "4",
              "--json",
            ],
            {
              cwd : repoRoot,
              env : {
                ...process.env,
                NO_COLOR : "1",
              },
            },
        );

        await testInfo.attach("stdout-native-ternary-safetensors", {
          body : stdout.trim() || "(empty)",
          contentType : "text/plain",
        });
        await testInfo.attach("stderr-native-ternary-safetensors", {
          body : stderr.trim() || "(empty)",
          contentType : "text/plain",
        });

        expect(stderr.trim()).toBe("");
        const payload = JSON.parse(stdout) as Record<string, unknown>;
        expect(payload).toMatchObject({
          family : "ternary",
          model : "toy-ternary",
          asset_format : "safetensors",
          asset_path : expect.stringContaining("toy-ternary.safetensors"),
          prompt_tokens : [ "hi" ],
          weight_dtype : "int4",
          dequant_path : "groupwise-int4",
          generated_tokens : expect.any(Array),
        });
        expect((payload.generated_tokens as unknown[]).length)
            .toBeGreaterThanOrEqual(
                4,
            );
      });
});
