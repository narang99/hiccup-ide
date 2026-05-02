export type Direction = "LR" | "TB";


export const toggleDirection = (direction: Direction) => {
    return direction === "LR" ? "TB" : "LR";
}