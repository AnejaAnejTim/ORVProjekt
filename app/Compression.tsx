export class BinWriter {
    k: number = 0;
    x: number = 0;
    writtenBytes: number = 0;
    buffer: number[] = [];

    writeFullByte(v: number) {
        const c = v & 0xff;
        this.writtenBytes++;
        this.buffer.push(c);
    }

    writeByte(byte: number) {
        for (let i = 7; i >= 0; i--) {
            const b = ((byte >> i) & 1) === 1;
            this.writeBit(b);
        }
    }

    writeBit(b: boolean) {
        this.x = (this.x << 1) | (b ? 1 : 0);
        this.k++;
        if (this.k === 8) {
            this.writeFullByte(this.x);
            this.k = 0;
            this.x = 0;
        }
    }

    closeFile(): Uint8Array {
        if (this.k > 0) {
            this.x <<= (8 - this.k);
            this.writeFullByte(this.x);
        }
        return Uint8Array.from(this.buffer);
    }
}

export function predictions(P: number[][], height: number, width: number): number[] {
    const errors = new Array(height * width);

    for (let x = 0; x < height; x++) {
        for (let y = 0; y < width; y++) {
            let prediction = 0;

            if (y === 0 && x === 0) {
                errors[y * height + x] = P[x][y];
            } else if (y === 0) {
                errors[y * height + x] = P[x - 1][y] - P[x][y];
            } else if (x === 0) {
                errors[y * height + x] = P[x][y - 1] - P[x][y];
            } else {
                const a = P[x - 1][y];
                const b = P[x][y - 1];
                const c = P[x - 1][y - 1];

                if (c >= Math.max(a, b)) {
                    prediction = Math.min(a, b);
                } else if (c <= Math.min(a, b)) {
                    prediction = Math.max(a, b);
                } else {
                    prediction = a + b - c;
                }

                errors[y * height + x] = prediction - P[x][y];
            }
        }
    }

    return errors;
}

export function IC(bw: BinWriter, C: number[], L: number, H: number) {
    if (H - L > 1) {
        if (C[H] !== C[L]) {
            const m = Math.floor((H + L) / 2);
            const g = Math.ceil(Math.log2(C[H] - C[L] + 1));
            const val = C[m] - C[L];

            for (let i = g - 1; i >= 0; i--) {
                bw.writeBit(((val >> i) & 1) === 1);
            }

            if (L < m) IC(bw, C, L, m);
            if (m < H) IC(bw, C, m, H);
        }
    }
}

export function compressRGBChannel(bw: BinWriter, P: number[][], height: number, width: number) {
    const predictionArray = predictions(P, height, width);

    const N = new Array(predictionArray.length);
    N[0] = predictionArray[0];
    for (let i = 1; i < predictionArray.length; i++) {
        if (predictionArray[i] >= 0) N[i] = 2 * predictionArray[i];
        else N[i] = 2 * Math.abs(predictionArray[i]) - 1;
    }

    const C = new Array(N.length);
    C[0] = N[0];
    for (let i = 1; i < C.length; i++) {
        C[i] = C[i - 1] + N[i];
    }

    for (let i = 15; i >= 0; i--) bw.writeBit(((height >> i) & 1) === 1);
    for (let i = 7; i >= 0; i--) bw.writeBit(((C[0] >> i) & 1) === 1);
    for (let i = 31; i >= 0; i--) bw.writeBit(((C[C.length - 1] >> i) & 1) === 1);
    for (let i = 31; i >= 0; i--) bw.writeBit(((C.length >> i) & 1) === 1);

    IC(bw, C, 0, C.length - 1);
}


export function compressImageRGB(
    R: number[][],
    G: number[][],
    B: number[][]
): Uint8Array {
    const height = R.length;
    const width = R[0].length;

    const bw = new BinWriter();

    compressRGBChannel(bw, R, height, width);
    compressRGBChannel(bw, G, height, width);
    compressRGBChannel(bw, B, height, width);

    return bw.closeFile();
}